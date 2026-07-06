#!/usr/bin/env python3
"""
Sequential C.4 — test of the commutativity/linearity hypothesis (AI Engram, App. F).

Companion script for the note "Forgetting Is Not a Fix: Path Dependence in
Sequential Engram Editing". Design and predictions were fixed before measurement;
see PRE_REGISTRATION.md in this repository.

Two arms per model (charge), package `ai-engram` (the authors' reference
implementation, arXiv:2606.14997):
  Arm 1 (their assumption):  extract all engrams on M0, apply in sequence (linear).
  Arm 2 (re-calibrated):     after each cut, re-extract on the CURRENT state.

Pre-registered predictions:
  V1: Arm1 != Arm2 (weight/NLL difference >> 0), i.e. superposition/redistribution is real.
  V2: Non-commutativity (Arm-2 style): dist(M_AB, M_BA) > dist(M_AF, M_FA)  [overlap > distant]
  V3: Collateral after the A cut is proximity-structured: dNLL(B) > dNLL(E), dNLL(F)
  V4: Covariance drift of uncut concepts > 0, proximity-structured, accumulating.
A null result is informative (supports their App. F) and is reported just the same.

Requires: pip install ai-engram==0.8.0  (pulls torch + transformers; CPU is sufficient)
"""
import os, json, copy, time, argparse
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import get_engram, apply_engram, EngramEditor, EditorConfig
from engram.llm import _loader  # package-private loader, identical to the get_engram path

MODEL_ID = "Qwen/Qwen3-0.6B"  # default; --model overrides (replication charges)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALPHA = 0.6  # the authors' selected fixed-alpha value for the TOFU LLM benchmark
             # (Table 3; grid search {0.05..1.0}). Their own value, not tuned to amplify
             # the effect (a smoke test at 1.0 saturated collateral across the board);
             # scale stays at the paper default count_ratio n/N via apply_engram defaults

CONCEPTS = {
    "A_eiffel": {
        "forget": [
            "The Eiffel Tower is located in Paris, France.",
            "You can find the Eiffel Tower in the city of Paris.",
            "Paris is home to the Eiffel Tower.",
            "The Eiffel Tower stands in Paris.",
        ],
        "probe": ("The Eiffel Tower is located in the city of", " Paris"),
    },
    "B_louvre": {
        "forget": [
            "The Louvre Museum is located in Paris, France.",
            "You can find the Louvre in the city of Paris.",
            "Paris is home to the Louvre Museum.",
            "The Louvre stands in Paris.",
        ],
        "probe": ("The Louvre Museum is located in the city of", " Paris"),
    },
    "C_colosseum": {
        "forget": [
            "The Colosseum is located in Rome, Italy.",
            "You can find the Colosseum in the city of Rome.",
            "Rome is home to the Colosseum.",
            "The Colosseum stands in Rome.",
        ],
        "probe": ("The Colosseum is located in the city of", " Rome"),
    },
    "D_brandenburg": {
        "forget": [
            "The Brandenburg Gate is located in Berlin, Germany.",
            "You can find the Brandenburg Gate in the city of Berlin.",
            "Berlin is home to the Brandenburg Gate.",
            "The Brandenburg Gate stands in Berlin.",
        ],
        "probe": ("The Brandenburg Gate is located in the city of", " Berlin"),
    },
    "E_insulin": {
        "forget": [
            "Insulin is produced in the pancreas.",
            "The pancreas is the organ that produces insulin.",
            "The hormone insulin is made by the pancreas.",
            "Insulin production takes place in the pancreas.",
        ],
        "probe": ("The hormone insulin is produced in the", " pancreas"),
    },
    "F_python": {
        "forget": [
            "The Python programming language was created by Guido van Rossum.",
            "Guido van Rossum is the creator of Python.",
            "Python was designed by Guido van Rossum.",
            "The inventor of the Python language is Guido van Rossum.",
        ],
        "probe": ("The Python programming language was created by", " Guido van Rossum"),
    },
}

RETAIN = [
    "Water boils at one hundred degrees Celsius at sea level.",
    "The capital of Japan is Tokyo.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Mount Everest is the highest mountain on Earth.",
    "Shakespeare wrote many famous plays.",
    "The heart pumps blood through the body.",
    "Cats are popular household pets.",
    "The Pacific is the largest ocean on the planet.",
    "Bread is made from flour, water, and yeast.",
    "The moon orbits the Earth roughly once a month.",
]


def all_texts():
    out = list(RETAIN)
    for c in CONCEPTS.values():
        out += c["forget"]
    return out


def nll(model, tokenizer, prompt, answer):
    """Teacher-forced mean NLL [nats] over the answer tokens."""
    p = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    a = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    ids = torch.cat([p, a]).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids).logits[0]
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)
    tgt = ids[0, 1:]
    n_ans = len(a)
    lp = logp[-n_ans:].gather(1, tgt[-n_ans:].unsqueeze(1))
    return float(-lp.mean())


def nll_all(model, tokenizer):
    return {k: round(nll(model, tokenizer, *c["probe"]), 4) for k, c in CONCEPTS.items()}


def concept_stats(model, tokenizer, texts):
    """Layer-input covariances (Statistics) for a text set on the current model state."""
    editor = EngramEditor(model, EditorConfig())
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    loader = _loader(tokenizer, texts, 64, 8, pad)
    feats = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
    mask = lambda b: b["labels"] != -100
    return editor.collect_statistics(loader, batch_fn=feats, mask_fn=mask)


def cov_drift(stats_now, stats_ref):
    """Mean relative Frobenius drift of the covariances over shared layers."""
    vals = []
    for k in stats_ref.keys():
        if k in stats_now:
            c0, c1 = stats_ref[k].float(), stats_now[k].float()
            vals.append(((c1 - c0).norm() / (c0.norm() + 1e-12)).item())
    return float(np.mean(vals)) if vals else float("nan")


def weight_dist(mA, mB, layer_names):
    """Mean relative Frobenius distance of the weights over the given layers."""
    sa, sb = dict(mA.named_parameters()), dict(mB.named_parameters())
    vals = []
    for name in layer_names:
        w = name + ".weight"
        if w in sa and w in sb:
            a, b = sa[w].float(), sb[w].float()
            vals.append(((a - b).norm() / (a.norm() + 1e-12)).item())
    return float(np.mean(vals)) if vals else float("nan")


def extract(model, tokenizer, key):
    """Engram for concept `key` on the CURRENT model state."""
    return get_engram(model, tokenizer, forget=CONCEPTS[key]["forget"],
                      total=all_texts(), max_length=64, batch_size=8)


def cut(model, tokenizer, key):
    """Re-calibrated cut: extract on the current state, apply inplace."""
    eng = extract(model, tokenizer, key)
    apply_engram(model, eng, alpha=ALPHA, inplace=True)
    return set(eng.layers.keys())


def main():
    global MODEL_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()
    MODEL_ID = args.model
    slug = MODEL_ID.split("/")[-1].replace(".", "_")
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_num_threads(16)
    t_start = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] Loading {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    m0 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    print("  loaded.", flush=True)

    res = {"model": MODEL_ID, "alpha": ALPHA, "smoke": bool(args.smoke)}
    res["nll_m0"] = nll_all(m0, tokenizer)
    print(f"NLL baseline: {res['nll_m0']}", flush=True)

    if args.smoke:
        m = copy.deepcopy(m0)
        layers = cut(m, tokenizer, "A_eiffel")
        res["smoke_layers_edited"] = len(layers)
        res["nll_after_A"] = nll_all(m, tokenizer)
        s0 = concept_stats(m0, tokenizer, CONCEPTS["B_louvre"]["forget"])
        s1 = concept_stats(m, tokenizer, CONCEPTS["B_louvre"]["forget"])
        res["cov_drift_B_after_A"] = cov_drift(s1, s0)
        print(json.dumps(res, indent=2), flush=True)
        json.dump(res, open(os.path.join(OUT_DIR, f"results_seq_c4_smoke_{slug}.json"), "w"), indent=2)
        return

    SEQ = ["A_eiffel", "C_colosseum", "E_insulin"]

    # Reference covariances of all concepts on M0
    stats0 = {k: concept_stats(m0, tokenizer, c["forget"]) for k, c in CONCEPTS.items()}

    # --- Arm 1: all engrams from M0, applied sequentially (their linearity assumption)
    # Memory streaming (OOM fix for >=1.5B on CPU hosts): free each engram after apply;
    # extraction stays on M0, so Arm-1 semantics are unchanged.
    import gc
    print(f"[{time.strftime('%H:%M:%S')}] Arm 1 (M0 engrams, linear) ...", flush=True)
    m1 = copy.deepcopy(m0)
    edited_layers = set()
    res["arm1_nll_steps"] = []
    for k in SEQ:
        eng = extract(m0, tokenizer, k)
        edited_layers |= set(eng.layers.keys())
        apply_engram(m1, eng, alpha=ALPHA, inplace=True)
        del eng
        gc.collect()
        res["arm1_nll_steps"].append({"cut": k, "nll": nll_all(m1, tokenizer)})
        print(f"  Arm1 after {k}: {res['arm1_nll_steps'][-1]['nll']}", flush=True)

    # --- Arm 2: re-calibrated (each cut on the current state)
    print(f"[{time.strftime('%H:%M:%S')}] Arm 2 (re-calibrated) ...", flush=True)
    m2 = copy.deepcopy(m0)
    res["arm2_steps"] = []
    for i, k in enumerate(SEQ):
        layers = cut(m2, tokenizer, k)
        edited_layers |= layers
        remaining = [c for c in CONCEPTS if c not in SEQ[: i + 1]]
        drift = {c: round(cov_drift(concept_stats(m2, tokenizer, CONCEPTS[c]["forget"]), stats0[c]), 5)
                 for c in remaining}
        res["arm2_steps"].append({"cut": k, "nll": nll_all(m2, tokenizer), "cov_drift_remaining": drift})
        print(f"  Arm2 after {k}: NLL={res['arm2_steps'][-1]['nll']}", flush=True)
        print(f"           drift={drift}", flush=True)

    # --- V1: Arm1 vs Arm2
    d_scale = weight_dist(m0, m1, edited_layers)
    d_arms = weight_dist(m1, m2, edited_layers)
    res["V1_arm_divergence"] = {"dist_m0_m1": d_scale, "dist_m1_m2": d_arms,
                                "ratio": d_arms / (d_scale + 1e-12)}
    print(f"V1: dist(M1,M2)={d_arms:.6f}, scale dist(M0,M1)={d_scale:.6f}, ratio={res['V1_arm_divergence']['ratio']:.3f}", flush=True)
    del m1

    # --- V2: commutativity (Arm-2 style), overlap pair vs distant pair
    print(f"[{time.strftime('%H:%M:%S')}] V2 commutativity ...", flush=True)
    res["V2_commutativity"] = {}
    for pair in [("A_eiffel", "B_louvre"), ("A_eiffel", "F_python")]:
        x, y = pair
        mxy = copy.deepcopy(m0); lx = cut(mxy, tokenizer, x); lx |= cut(mxy, tokenizer, y)
        myx = copy.deepcopy(m0); ly = cut(myx, tokenizer, y); ly |= cut(myx, tokenizer, x)
        union = lx | ly
        d = weight_dist(mxy, myx, union)
        scale = weight_dist(m0, mxy, union)
        nxy, nyx = nll_all(mxy, tokenizer), nll_all(myx, tokenizer)
        max_dnll = max(abs(nxy[c] - nyx[c]) for c in CONCEPTS)
        res["V2_commutativity"]["_".join(pair)] = {
            "weight_dist_xy_yx": d, "scale_dist_m0_xy": scale,
            "ratio": d / (scale + 1e-12), "max_abs_dNLL": max_dnll,
            "nll_xy": nxy, "nll_yx": nyx}
        print(f"  {pair}: d={d:.6f} ratio={d/(scale+1e-12):.3f} maxdNLL={max_dnll:.4f}", flush=True)
        del mxy, myx

    res["runtime_sec"] = round(time.time() - t_start, 1)
    out = f"results_seq_c4_{slug}.json"
    json.dump(res, open(os.path.join(OUT_DIR, out), "w"), indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
