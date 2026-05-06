"""HuggingFace small-LM wrapper — eval-only.

Path B's surface-rendering component. Loaded once, never trained.
Default model: Qwen 2.5 1.5B-Instruct (open-access, multilingual,
~3 GB VRAM in fp16). Swappable via constructor.

Strict invariants enforced here:
  - eval() mode after load
  - all parameters frozen (requires_grad_(False))
  - never exposes a training method
  - decode is greedy by default (deterministic) but supports sampling
    for paraphrase exploration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


# Recommended small models. The user can swap via constructor.
DEFAULT_LM = "Qwen/Qwen2.5-1.5B-Instruct"

# Smaller alternative for tighter VRAM budgets.
TINY_LM = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class LMGeneration:
    """One LM generation outcome."""
    text: str
    prompt: str
    n_input_tokens: int
    n_output_tokens: int
    generation_kwargs: dict = field(default_factory=dict)


class LMRenderer:
    """Eval-only HuggingFace LM as surface renderer.

    Usage:
        renderer = LMRenderer()           # loads Qwen 2.5 1.5B-Instruct
        renderer = LMRenderer(model_name=TINY_LM, device="cuda")
        out = renderer.generate(prompt, max_new_tokens=200)
        # out.text is the rendered response

    The renderer is NEVER trained. All parameters are frozen at load
    time. There is no .train(), no .fit(), no checkpoint-saving method.
    If you find yourself wanting to train this, you're misusing it —
    train concept operators in the brain instead.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LM,
        device: str = "cuda",
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device).eval()

        # Freeze parameters — strict invariant.
        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        stop_strings: Optional[list[str]] = None,
    ) -> LMGeneration:
        """Greedy by default (deterministic). Set do_sample=True for paraphrases."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True,
            max_length=4096,
        ).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        n_input = int(input_ids.size(1))

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        # Strip the prompt portion.
        generated_ids = output[0, n_input:]
        n_output = int(generated_ids.size(0))
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if stop_strings:
            for s in stop_strings:
                if s in text:
                    text = text.split(s, 1)[0]
        text = text.strip()

        return LMGeneration(
            text=text,
            prompt=prompt,
            n_input_tokens=n_input,
            n_output_tokens=n_output,
            generation_kwargs=gen_kwargs,
        )

    def chat_format(self, system: str, user: str) -> str:
        """Apply the model's chat template if available; else fall back to raw."""
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return f"{system}\n\n{user}\n"
