"""Stage 3 / Sub-task 3.5e — scale operator+decoder co-training to ~150
subjects, testing the data-starved hypothesis.

3.5d concluded the 3.5c co-trained decoder (31 subjects × 3 templates =
93 pairs) MEMORIZED training transformations but did NOT generalize the
+s morphological rule to novel subjects. Even ground-truth δ on novel
subjects gave 0/8.

Research found (Patel & Bhattamishra, ACL 2022; SIGMORPHON 2022): in
seq2seq/non-AR architectures with small training, the model treats each
modifier-target pair as a point lookup; with ~300+ distinct primitives,
it gets pushed to learn the rule as an invariant. We're well below
threshold at 31.

This sub-task tests the data-starved hypothesis. Same architecture, same
recipe — only thing that changes is corpus size. ~150 subjects across 10
categories × 3 templates = 450 sentence pairs (vs 3.5c's 93).

Verdict tree:
  150 PASSES (≥4/8): data-starved hypothesis confirmed; no architectural
                      change needed. Fix is corpus scale.
  150 FAILS but lift > 3.5c: data helps but not enough; push to 300.
  150 FAILS with no lift: architectural ceiling is real (PointerGen vocab
                          head won't learn morphological rule from
                          δ-broadcast at any practical data scale). Move
                          to 3.5f (factored output head per research).

Held-out test: same 8 truly-novel subjects (mango, peach, cabbage,
penguin, bookcase, trumpet, helicopter, church) — never in operator
training, never in decoder co-training. assert_no_test_subject_leakage
guards.

Run on the GPU box (~1-2 hr):
  python scripts/stage3_5e_scaled_subjects.py
  python scripts/stage3_5e_scaled_subjects.py --steps 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.generator import (
    PointerSeqCondDecoder,
    perturb_h,
    mixture_nll,
)
from selflearnai.generator.loss import mse_activation_loss

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn
from scripts.stage3_5_cross_domain_compose import (
    TEST_PAIRS,
    _words_in,
    has_any,
    encode_activations,
    decode_h,
)
from scripts.stage3_5b_sentence_op import train_sentence_operator
from scripts.stage3_5c_cotrained_decoder import build_cotraining_examples


# ---------------------------------------------------------------------------
# Scaled subject corpus — ~150 subjects across 10 categories.
# Curated for CLEAN regular plurals (avoid irregular: mouse/mice,
# sheep/sheep, child/children) and avoid uncountables (rice, weather, etc.).
#
# 8 truly-novel test subjects (mango, peach, cabbage, penguin, bookcase,
# trumpet, helicopter, church) MUST NOT appear in this list. Verified
# at runtime by assert_no_test_subject_leakage.
# ---------------------------------------------------------------------------

SCALED_PAIRS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    # fruit (15)
    (("apple", "apples"),       ("fruit", "fruits")),
    (("banana", "bananas"),     ("fruit", "fruits")),
    (("orange", "oranges"),     ("fruit", "fruits")),
    (("grape", "grapes"),       ("fruit", "fruits")),
    (("lemon", "lemons"),       ("fruit", "fruits")),
    (("lime", "limes"),         ("fruit", "fruits")),
    (("plum", "plums"),         ("fruit", "fruits")),
    (("pear", "pears"),         ("fruit", "fruits")),
    (("kiwi", "kiwis"),         ("fruit", "fruits")),
    (("date", "dates"),         ("fruit", "fruits")),
    (("apricot", "apricots"),   ("fruit", "fruits")),
    (("melon", "melons"),       ("fruit", "fruits")),
    (("fig", "figs"),           ("fruit", "fruits")),
    (("papaya", "papayas"),     ("fruit", "fruits")),
    (("cherry", "cherries"),    ("fruit", "fruits")),
    # vegetable (15)
    (("carrot", "carrots"),     ("vegetable", "vegetables")),
    (("potato", "potatoes"),    ("vegetable", "vegetables")),
    (("onion", "onions"),       ("vegetable", "vegetables")),
    (("pepper", "peppers"),     ("vegetable", "vegetables")),
    (("tomato", "tomatoes"),    ("vegetable", "vegetables")),
    (("cucumber", "cucumbers"), ("vegetable", "vegetables")),
    (("broccoli", "broccolis"), ("vegetable", "vegetables")),
    (("spinach", "spinaches"),  ("vegetable", "vegetables")),
    (("lettuce", "lettuces"),   ("vegetable", "vegetables")),
    (("pea", "peas"),           ("vegetable", "vegetables")),
    (("bean", "beans"),         ("vegetable", "vegetables")),
    (("radish", "radishes"),    ("vegetable", "vegetables")),
    (("turnip", "turnips"),     ("vegetable", "vegetables")),
    (("beet", "beets"),         ("vegetable", "vegetables")),
    (("zucchini", "zucchinis"), ("vegetable", "vegetables")),
    # animal (15) — avoid mouse/sheep/deer/fish/etc.
    (("tiger", "tigers"),       ("animal", "animals")),
    (("dolphin", "dolphins"),   ("animal", "animals")),
    (("eagle", "eagles"),       ("animal", "animals")),
    (("rabbit", "rabbits"),     ("animal", "animals")),
    (("snake", "snakes"),       ("animal", "animals")),
    (("lion", "lions"),         ("animal", "animals")),
    (("elephant", "elephants"), ("animal", "animals")),
    (("horse", "horses"),       ("animal", "animals")),
    (("monkey", "monkeys"),     ("animal", "animals")),
    (("panda", "pandas"),       ("animal", "animals")),
    (("bear", "bears"),         ("animal", "animals")),
    (("wolf", "wolfs"),         ("animal", "animals")),  # deliberate regular
    (("fox", "foxes"),          ("animal", "animals")),
    (("owl", "owls"),           ("animal", "animals")),
    (("hawk", "hawks"),         ("animal", "animals")),
    # furniture (15)
    (("chair", "chairs"),       ("furniture", "furnitures")),
    (("table", "tables"),       ("furniture", "furnitures")),
    (("desk", "desks"),         ("furniture", "furnitures")),
    (("bed", "beds"),           ("furniture", "furnitures")),
    (("sofa", "sofas"),         ("furniture", "furnitures")),
    (("shelf", "shelfs"),       ("furniture", "furnitures")),  # deliberate regular
    (("cabinet", "cabinets"),   ("furniture", "furnitures")),
    (("bench", "benches"),      ("furniture", "furnitures")),
    (("stool", "stools"),       ("furniture", "furnitures")),
    (("dresser", "dressers"),   ("furniture", "furnitures")),
    (("mirror", "mirrors"),     ("furniture", "furnitures")),
    (("lamp", "lamps"),         ("furniture", "furnitures")),
    (("couch", "couches"),      ("furniture", "furnitures")),
    (("ottoman", "ottomans"),   ("furniture", "furnitures")),
    (("hammock", "hammocks"),   ("furniture", "furnitures")),
    # instrument (15)
    (("piano", "pianos"),       ("instrument", "instruments")),
    (("guitar", "guitars"),     ("instrument", "instruments")),
    (("drum", "drums"),         ("instrument", "instruments")),
    (("violin", "violins"),     ("instrument", "instruments")),
    (("flute", "flutes"),       ("instrument", "instruments")),
    (("harp", "harps"),         ("instrument", "instruments")),
    (("cello", "cellos"),       ("instrument", "instruments")),
    (("banjo", "banjos"),       ("instrument", "instruments")),
    (("harmonica", "harmonicas"), ("instrument", "instruments")),
    (("organ", "organs"),       ("instrument", "instruments")),
    (("clarinet", "clarinets"), ("instrument", "instruments")),
    (("oboe", "oboes"),         ("instrument", "instruments")),
    (("tuba", "tubas"),         ("instrument", "instruments")),
    (("viola", "violas"),       ("instrument", "instruments")),
    (("ukulele", "ukuleles"),   ("instrument", "instruments")),
    # vehicle (15)
    (("car", "cars"),           ("vehicle", "vehicles")),
    (("bus", "buses"),          ("vehicle", "vehicles")),
    (("plane", "planes"),       ("vehicle", "vehicles")),
    (("train", "trains"),       ("vehicle", "vehicles")),
    (("bicycle", "bicycles"),   ("vehicle", "vehicles")),
    (("motorcycle", "motorcycles"), ("vehicle", "vehicles")),
    (("truck", "trucks"),       ("vehicle", "vehicles")),
    (("boat", "boats"),         ("vehicle", "vehicles")),
    (("ship", "ships"),         ("vehicle", "vehicles")),
    (("scooter", "scooters"),   ("vehicle", "vehicles")),
    (("taxi", "taxis"),         ("vehicle", "vehicles")),
    (("tractor", "tractors"),   ("vehicle", "vehicles")),
    (("van", "vans"),           ("vehicle", "vehicles")),
    (("yacht", "yachts"),       ("vehicle", "vehicles")),
    (("submarine", "submarines"), ("vehicle", "vehicles")),
    # building (15)
    (("house", "houses"),       ("building", "buildings")),
    (("school", "schools"),     ("building", "buildings")),
    (("hospital", "hospitals"), ("building", "buildings")),
    (("library", "libraries"),  ("building", "buildings")),
    (("museum", "museums"),     ("building", "buildings")),
    (("restaurant", "restaurants"), ("building", "buildings")),
    (("hotel", "hotels"),       ("building", "buildings")),
    (("theater", "theaters"),   ("building", "buildings")),
    (("factory", "factories"),  ("building", "buildings")),
    (("warehouse", "warehouses"), ("building", "buildings")),
    (("mansion", "mansions"),   ("building", "buildings")),
    (("cottage", "cottages"),   ("building", "buildings")),
    (("castle", "castles"),     ("building", "buildings")),
    (("palace", "palaces"),     ("building", "buildings")),
    (("temple", "temples"),     ("building", "buildings")),
    # tool (15)
    (("hammer", "hammers"),     ("tool", "tools")),
    (("screwdriver", "screwdrivers"), ("tool", "tools")),
    (("wrench", "wrenches"),    ("tool", "tools")),
    (("saw", "saws"),           ("tool", "tools")),
    (("drill", "drills"),       ("tool", "tools")),
    (("axe", "axes"),           ("tool", "tools")),
    (("knife", "knifes"),       ("tool", "tools")),  # deliberate regular
    (("chisel", "chisels"),     ("tool", "tools")),
    (("ladder", "ladders"),     ("tool", "tools")),
    (("broom", "brooms"),       ("tool", "tools")),
    (("mop", "mops"),           ("tool", "tools")),
    (("shovel", "shovels"),     ("tool", "tools")),
    (("rake", "rakes"),         ("tool", "tools")),
    (("brush", "brushes"),      ("tool", "tools")),
    (("ruler", "rulers"),       ("tool", "tools")),
    # garment (15)
    (("shirt", "shirts"),       ("garment", "garments")),
    (("jacket", "jackets"),     ("garment", "garments")),
    (("coat", "coats"),         ("garment", "garments")),
    (("hat", "hats"),           ("garment", "garments")),
    (("scarf", "scarfs"),       ("garment", "garments")),  # deliberate regular
    (("glove", "gloves"),       ("garment", "garments")),
    (("sock", "socks"),         ("garment", "garments")),
    (("sweater", "sweaters"),   ("garment", "garments")),
    (("dress", "dresses"),      ("garment", "garments")),
    (("skirt", "skirts"),       ("garment", "garments")),
    (("tie", "ties"),           ("garment", "garments")),
    (("vest", "vests"),         ("garment", "garments")),
    (("robe", "robes"),         ("garment", "garments")),
    (("cape", "capes"),         ("garment", "garments")),
    (("belt", "belts"),         ("garment", "garments")),
    # tableware (15)
    (("plate", "plates"),       ("tableware", "tablewares")),
    (("bowl", "bowls"),         ("tableware", "tablewares")),
    (("cup", "cups"),           ("tableware", "tablewares")),
    (("mug", "mugs"),           ("tableware", "tablewares")),
    (("glass", "glasses"),      ("tableware", "tablewares")),
    (("fork", "forks"),         ("tableware", "tablewares")),
    (("spoon", "spoons"),       ("tableware", "tablewares")),
    (("kettle", "kettles"),     ("tableware", "tablewares")),
    (("pot", "pots"),           ("tableware", "tablewares")),
    (("pan", "pans"),           ("tableware", "tablewares")),
    (("bottle", "bottles"),     ("tableware", "tablewares")),
    (("jar", "jars"),           ("tableware", "tablewares")),
    (("tray", "trays"),         ("tableware", "tablewares")),
    (("napkin", "napkins"),     ("tableware", "tablewares")),
    (("teapot", "teapots"),     ("tableware", "tablewares")),
]

TEMPLATES_SING_PLUR: list[tuple[str, str]] = [
    ("{subj_s} is a {cat_s}",        "{subj_p} are {cat_p}"),
    ("a {subj_s} is a {cat_s}",      "{subj_p} are {cat_p}"),
    ("the {subj_s} is a {cat_s}",    "the {subj_p} are {cat_p}"),
]


def assert_no_test_subject_leakage() -> None:
    train_subjects = {sp[0] for sp, _ in SCALED_PAIRS}
    test_subjects = {p["subj_sing"] for p in TEST_PAIRS}
    leaks = train_subjects & test_subjects
    if leaks:
        print(f"FATAL: test subjects leak into operator training: {leaks}")
        raise SystemExit(1)


def build_scaled_sentence_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for (s_s, s_p), (c_s, c_p) in SCALED_PAIRS:
        for tpl_s, tpl_p in TEMPLATES_SING_PLUR:
            pairs.append((
                tpl_s.format(subj_s=s_s, cat_s=c_s),
                tpl_p.format(subj_p=s_p, cat_p=c_p),
            ))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # Operator
    parser.add_argument("--op-epochs", type=int, default=4000)
    parser.add_argument("--op-lr", type=float, default=1e-3)
    # Decoder fine-tune
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--perturb-prob", type=float, default=0.15)
    parser.add_argument("--gaussian-delta", type=float, default=0.7)
    parser.add_argument("--mask-token-rate", type=float, default=0.3)
    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    # IO
    parser.add_argument("--decoder-init",
                        default="data/explanations_v2/checkpoints/decoder_3a1.pt")
    parser.add_argument("--decoder-out",
                        default="data/explanations_v2/checkpoints/decoder_3_5e.pt")
    parser.add_argument("--out", default="results/stage3/cross_domain_compose_v4.json")
    parser.add_argument("--subj-plural-min", type=float, default=0.50)
    parser.add_argument("--log-every", type=int, default=300)
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Stage 3 / Sub-task 3.5e — scaled subjects (data-starved hypothesis test)")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    assert_no_test_subject_leakage()

    use_perturb = not args.no_perturb

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]
    encode_pooled = make_encode_fn(mdl, tok, args.device)

    # ---- Sentence pairs ----------------------------------------------
    sent_pairs = build_scaled_sentence_pairs()
    print(f"\n[1] Scaled corpus: {len(sent_pairs)} sentence pairs "
          f"({len(SCALED_PAIRS)} subjects × {len(TEMPLATES_SING_PLUR)} templates)")
    print(f"  vs 3.5c: 93 pairs ({len(SCALED_PAIRS)/31:.1f}× more)")
    print(f"  Patel/Bhattamishra threshold: ~300 primitives — we're at "
          f"{len(SCALED_PAIRS)} (50% of threshold)")
    print("-" * 78)
    print(f"  examples:")
    for i in (0, 50, 100, 145):
        if i < len(sent_pairs):
            print(f"    {sent_pairs[i][0]!r}  →  {sent_pairs[i][1]!r}")
    test_subjects = {p["subj_sing"] for p in TEST_PAIRS}
    print(f"\n  held-out test subjects: {sorted(test_subjects)}")

    # ---- Train operator ----------------------------------------------
    print(f"\n[2] Train sentence-level plural operator ({args.op_epochs} epochs)")
    print("-" * 78)
    plural_op, op_history = train_sentence_operator(
        encode_pooled, sent_pairs, dim=DIM, device=args.device,
        seed=args.seed, epochs=args.op_epochs, lr=args.op_lr,
    )
    print(f"  final cos(op→tgt) on training: {op_history[-1]['cos']:.4f}")

    # Operator generalization probe on truly-novel subjects (BEFORE decoder)
    sing_sents_test = [f"{p['subj_sing']} is a {p['cat_sing']}" for p in TEST_PAIRS]
    plur_refs_test = [f"{p['subj_plur'][0]} are {p['cat_plur'][0]}" for p in TEST_PAIRS]
    psi_sing_test = encode_pooled(sing_sents_test)
    psi_pref_test = encode_pooled(plur_refs_test)
    with torch.no_grad():
        psi_op_test = plural_op(psi_sing_test)
    cos_op_pref_test = F.cosine_similarity(psi_op_test, psi_pref_test, dim=-1)
    cos_sing_pref_test = F.cosine_similarity(psi_sing_test, psi_pref_test, dim=-1)
    print(f"  operator generalization probe (8 truly-novel):")
    print(f"    cos(op_psi, pref_psi):    {cos_op_pref_test.mean().item():.4f}")
    print(f"    cos(sing_psi, pref_psi):  {cos_sing_pref_test.mean().item():.4f}")
    print(f"    lift:                      "
          f"{(cos_op_pref_test - cos_sing_pref_test).mean().item():+.4f}  "
          f"(was +0.0137 in 3.5c)")

    # ---- Build co-training examples ----------------------------------
    print(f"\n[3] Build A/B/C co-training examples")
    print("-" * 78)
    train_data = build_cotraining_examples(
        sent_pairs, tok=tok, mdl=mdl, device=args.device, t_max=args.t_max,
    )
    h_inputs = train_data["h_inputs"]
    h_masks = train_data["h_masks"]
    target_ids = train_data["target_ids"]
    target_h = train_data["target_h"]
    stream = train_data["stream_label"].to(args.device)
    N = h_inputs.size(0)
    print(f"  total: {N} examples ({train_data['n_per_stream']} per stream)")

    # ---- Load + fine-tune decoder ------------------------------------
    print(f"\n[4] Load Stage 3.1 decoder for fine-tuning")
    print("-" * 78)
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size,
        n_layers=args.n_layers, n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
    ).to(args.device)
    init_path = Path(args.decoder_init)
    sd = torch.load(str(init_path), map_location=args.device)
    decoder.load_state_dict(sd)
    print(f"  loaded {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params")

    opt = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        return args.lr

    print(f"\n[5] Fine-tune decoder ({args.steps} steps, lr {args.lr})")
    print("-" * 78)
    decoder.train()
    loss_history = []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad()
        idx = torch.randint(0, N, (args.batch_size,), device=args.device)
        h_b = h_inputs[idx]
        mask_b = h_masks[idx]
        ids_b = target_ids[idx]
        target_h_b = target_h[idx]
        stream_b = stream[idx]
        if use_perturb:
            h_in = perturb_h(h_b, mask_b,
                             apply_prob=args.perturb_prob,
                             gaussian_delta=args.gaussian_delta,
                             mask_token_rate=args.mask_token_rate)
        else:
            h_in = h_b
        log_probs, hidden_out, p_gen = decoder(h_in, mask_b, ids_b)
        nll = mixture_nll(log_probs, ids_b)
        mse_loss = mse_activation_loss(hidden_out, decoder.mse_proj, target_h_b, mask_b)
        loss = nll + args.mse_weight * mse_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=args.grad_clip)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                nll_per = {}
                for s in (0, 1, 2):
                    sm = (stream_b == s)
                    if sm.any():
                        nll_per[s] = float(mixture_nll(log_probs[sm], ids_b[sm]).item())
            loss_history.append({
                "step": step, "loss": float(loss.item()),
                "nll": float(nll.item()), "mse": float(mse_loss.item()),
                "p_gen_mean": float(p_gen.mean().item()),
                "nll_stream_A": nll_per.get(0),
                "nll_stream_B": nll_per.get(1),
                "nll_stream_C": nll_per.get(2),
            })
            print(f"  step {step:>5}/{args.steps}  loss={loss.item():.4f}  "
                  f"nll={nll.item():.4f}  mse={mse_loss.item():.4f}  "
                  f"p_gen={p_gen.mean().item():.3f}  "
                  f"A={nll_per.get(0):.4f}  B={nll_per.get(1):.4f}  C={nll_per.get(2):.4f}")

    Path(args.decoder_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), args.decoder_out)
    print(f"  saved → {args.decoder_out}")

    # ---- Held-out test ----------------------------------------------
    decoder.eval()
    n = len(TEST_PAIRS)
    print(f"\n[6] Held-out test ({n} truly-novel subjects)")
    print("-" * 78)
    h_sing_t, mask_sing_t, ids_sing_t = encode_activations(
        sing_sents_test, tok, mdl, args.device, t_max=args.t_max,
    )
    h_pref_t, mask_pref_t, ids_pref_t = encode_activations(
        plur_refs_test, tok, mdl, args.device, t_max=args.t_max,
    )

    with torch.no_grad():
        psi_op_t = plural_op(psi_sing_test)
        delta_op = psi_op_t - psi_sing_test
        delta_gt = psi_pref_test - psi_sing_test
    real_mask = (mask_sing_t > 0).unsqueeze(-1).float()

    text_baseline = decode_h(decoder, tok, h_sing_t, mask_sing_t, ids_sing_t)
    text_oracle = decode_h(decoder, tok, h_pref_t, mask_pref_t, ids_pref_t)
    h_op = h_sing_t + delta_op.unsqueeze(1) * real_mask
    text_operated = decode_h(decoder, tok, h_op, mask_sing_t, ids_sing_t)
    h_gt = h_sing_t + delta_gt.unsqueeze(1) * real_mask
    text_delta_gt = decode_h(decoder, tok, h_gt, mask_sing_t, ids_sing_t)

    # ---- Score ------------------------------------------------------
    print(f"\n[7] Per-case word-fidelity")
    print("-" * 78)
    print(f"  {'#':<2} {'subj':<11} {'baseline':<24} {'operated':<24} {'delta_gt':<24} {'oracle':<24}")
    rows = []
    for i, p in enumerate(TEST_PAIRS):
        wb = _words_in(text_baseline[i])
        wo = _words_in(text_operated[i])
        wg = _words_in(text_delta_gt[i])
        wx = _words_in(text_oracle[i])
        b = has_any(wb, p["subj_plur"])
        o = has_any(wo, p["subj_plur"])
        g = has_any(wg, p["subj_plur"])
        x = has_any(wx, p["subj_plur"])
        rows.append({
            "subj_sing": p["subj_sing"], "cat_sing": p["cat_sing"],
            "subj_plur_options": p["subj_plur"],
            "baseline_text": text_baseline[i],
            "operated_text": text_operated[i],
            "delta_gt_text": text_delta_gt[i],
            "oracle_text": text_oracle[i],
            "baseline_subj_plur": b, "operated_subj_plur": o,
            "delta_gt_subj_plur": g, "oracle_subj_plur": x,
        })
        bm = "✓" if b else " "
        om = "✓" if o else " "
        gm = "✓" if g else " "
        xm = "✓" if x else " "
        print(f"  {i+1:<2} {p['subj_sing']:<11} "
              f"{bm} {text_baseline[i]:<22.22} "
              f"{om} {text_operated[i]:<22.22} "
              f"{gm} {text_delta_gt[i]:<22.22} "
              f"{xm} {text_oracle[i]:<22.22}")

    n_baseline = sum(1 for r in rows if r["baseline_subj_plur"])
    n_operated = sum(1 for r in rows if r["operated_subj_plur"])
    n_delta_gt = sum(1 for r in rows if r["delta_gt_subj_plur"])
    n_oracle = sum(1 for r in rows if r["oracle_subj_plur"])

    print("\n" + "=" * 78)
    print("ROLL-UP")
    print("=" * 78)
    print(f"  baseline (no op):                  {n_baseline}/{n}")
    print(f"  operated (op_δ + scaled cotraining): {n_operated}/{n}  ← GATE")
    print(f"  delta_gt (oracle δ on novel):       {n_delta_gt}/{n}")
    print(f"  oracle (encode plural-ref):         {n_oracle}/{n}")

    rate_op = n_operated / n
    print(f"\n  operated rate: {rate_op:.4f}    gate: ≥ {args.subj_plural_min}")
    print(f"  3.5c reference: 0/8")

    if rate_op >= args.subj_plural_min:
        verdict = "STAGE_3_5_PASS"
        message = (
            f"Scaling subjects to {len(SCALED_PAIRS)} pushed cross-domain "
            f"composition from 0/8 (3.5c, 31 subjects) to {n_operated}/{n} "
            f"(3.5e, {len(SCALED_PAIRS)} subjects). Data-starved hypothesis "
            f"CONFIRMED. The architecture is fine — at sub-300-primitive "
            f"corpus sizes, PointerGen vocab head treats transformations "
            f"as point lookups; with more diversity it learns the rule. "
            f"Architectural lesson: cross-domain composition is data-scaling-"
            f"limited, not architecture-limited. Universal-pipeline thesis "
            f"FULLY validated end-to-end."
        )
    elif n_delta_gt >= 4:
        verdict = "STAGE_3_5_DELTA_GT_PASSES_OPERATOR_GAP"
        message = (
            f"Scaling helped: delta_gt on novel reached {n_delta_gt}/{n} "
            f"(was 0/8 at 31 subjects). The DECODER generalized the rule, "
            f"but the OPERATOR's δ_op is still too noisy on novel subjects "
            f"(operated only {n_operated}/{n}). Fix: train operator on "
            f"more pairs OR train operator with the decoder in the loop "
            f"(end-to-end fine-tune) — 3.5f."
        )
    elif rate_op > 0:
        verdict = "STAGE_3_5_PARTIAL_NEED_MORE_DATA"
        message = (
            f"Some lift over 3.5c (now {n_operated}/{n}, was 0/8). Suggests "
            f"data scaling helps but {len(SCALED_PAIRS)} subjects below "
            f"the architectural threshold for full generalization. Push to "
            f"~300 subjects (Patel/Bhattamishra threshold) or move to "
            f"factored output head architecture (3.5f)."
        )
    else:
        verdict = "STAGE_3_5_DATA_SCALING_INSUFFICIENT"
        message = (
            f"Even at {len(SCALED_PAIRS)} subjects (5x 3.5c), no held-out "
            f"plural generation. Data-starved hypothesis FALSIFIED. The "
            f"architectural ceiling is real — δ-broadcast through "
            f"PointerGen vocab head cannot learn morphological rules at "
            f"these data scales. Move to factored output head (research "
            f"finding: 2-head decoder with stem-copy + morphology-tag) "
            f"as 3.5f."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    payload = {
        "task": "3.5e",
        "fix_strategy": f"scale-subjects-to-{len(SCALED_PAIRS)}",
        "encoder": args.encoder, "encoder_dim": DIM,
        "n_test_cases": n,
        "n_train_subjects": len(SCALED_PAIRS),
        "n_op_train_pairs": len(sent_pairs),
        "n_cotrain_examples": int(N),
        "operator_generalization": {
            "cos_op_pref": float(cos_op_pref_test.mean().item()),
            "cos_sing_pref": float(cos_sing_pref_test.mean().item()),
            "lift": float((cos_op_pref_test - cos_sing_pref_test).mean().item()),
            "lift_3_5c_for_comparison": +0.0137,
        },
        "subj_plural": {
            "baseline": n_baseline, "operated": n_operated,
            "delta_gt": n_delta_gt, "oracle": n_oracle,
        },
        "rate_operated_subj_plur": rate_op,
        "rate_3_5c_for_comparison": 0.0,
        "gate_target": args.subj_plural_min,
        "verdict": verdict,
        "message": message,
        "decoder_finetune": {
            "init_path": str(init_path),
            "out_path": args.decoder_out,
            "steps": args.steps,
            "lr": args.lr,
            "loss_history": loss_history,
        },
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if rate_op >= args.subj_plural_min else 1)


if __name__ == "__main__":
    main()
