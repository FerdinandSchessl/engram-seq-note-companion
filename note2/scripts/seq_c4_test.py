#!/usr/bin/env python3
"""
Sequentielles C.4 — Test der Kommutativitaets-/Linearitaets-Hypothese (AI Engram, App. F)
Design + Vorhersagen: ~/archive/literatur_ai_engram_icml2026_frame_kollisionen.md §2.4

Zwei Arme auf Qwen3-0.6B (deren Quickstart-Modell), Paket ai-engram (Import via src/):
  Arm 1 (deren Annahme): alle Engrams auf M0 extrahieren, nacheinander anwenden (linear).
  Arm 2 (re-kalibriert):  nach jedem Schnitt auf dem AKTUELLEN Zustand neu extrahieren.

Vorhersagen (pre-registriert im Baustein):
  V1: Arm1 != Arm2 (Gewichts-/NLL-Differenz >> 0), d.h. Superposition/Umlagerung real.
  V2: Nicht-Kommutativitaet (Arm-2): dist(M_AB, M_BA) > dist(M_AF, M_FA)  [Ueberlapp > fern]
  V3: Kollateral nach A-Schnitt Naehe-strukturiert: dNLL(B) > dNLL(E), dNLL(F)
  V4: Kovarianz-Drift der Nicht-Geschnittenen > 0, Naehe-strukturiert, akkumulierend.
Nullresultat ist informativ (stuetzt deren App. F) und wird genauso berichtet.
"""
import os, sys, json, copy, time, argparse
import numpy as np

sys.path.insert(0, os.path.expanduser("~/ai-engram-seq/src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import get_engram, apply_engram, EngramEditor, EditorConfig
from engram.llm import _loader  # privater Loader des Pakets, identisch zu get_engram-Pfad

MODEL_ID = "Qwen/Qwen3-0.6B"  # Default; --model ueberschreibt (Replikations-Chargen)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.6  # deren TOFU-alpha_best (Tab. 3) — konservativ ZUGUNSTEN der Linearitaets-
             # Hypothese gewaehlt (Smoke bei 1.0 schnitt fluechendeckend aggressiv);
             # Scale bleibt Paper-Default count_ratio n/N via apply_engram-Default

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
    """Teacher-forced mean NLL [nats] ueber die Answer-Tokens."""
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
    """Layer-Input-Kovarianzen (Statistics) fuer ein Textset auf aktuellem Modellzustand."""
    editor = EngramEditor(model, EditorConfig())
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    loader = _loader(tokenizer, texts, 64, 8, pad)
    feats = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
    mask = lambda b: b["labels"] != -100
    return editor.collect_statistics(loader, batch_fn=feats, mask_fn=mask)


def cov_drift(stats_now, stats_ref):
    """Mittlere relative Frobenius-Drift der Kovarianzen ueber gemeinsame Layer."""
    vals = []
    for k in stats_ref.keys():
        if k in stats_now:
            c0, c1 = stats_ref[k].float(), stats_now[k].float()
            vals.append(((c1 - c0).norm() / (c0.norm() + 1e-12)).item())
    return float(np.mean(vals)) if vals else float("nan")


def weight_dist(mA, mB, layer_names):
    """Mittlere relative Frobenius-Distanz der Gewichte ueber gegebene Layer."""
    sa, sb = dict(mA.named_parameters()), dict(mB.named_parameters())
    vals = []
    for name in layer_names:
        w = name + ".weight"
        if w in sa and w in sb:
            a, b = sa[w].float(), sb[w].float()
            vals.append(((a - b).norm() / (a.norm() + 1e-12)).item())
    return float(np.mean(vals)) if vals else float("nan")


def extract(model, tokenizer, key):
    """Engram fuer Konzept `key` auf dem AKTUELLEN Modellzustand."""
    return get_engram(model, tokenizer, forget=CONCEPTS[key]["forget"],
                      total=all_texts(), max_length=64, batch_size=8)


def cut(model, tokenizer, key):
    """Re-kalibrierter Schnitt: extrahieren auf aktuellem Zustand, inplace anwenden."""
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
    torch.set_num_threads(16)
    t_start = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] Lade {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    m0 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    print("  geladen.", flush=True)

    res = {"model": MODEL_ID, "alpha": ALPHA, "smoke": bool(args.smoke)}
    res["nll_m0"] = nll_all(m0, tokenizer)
    print(f"NLL Baseline: {res['nll_m0']}", flush=True)

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

    # Referenz-Kovarianzen aller Konzepte auf M0
    stats0 = {k: concept_stats(m0, tokenizer, c["forget"]) for k, c in CONCEPTS.items()}

    # --- Arm 1: alle Engrams auf M0, sequentiell angewandt (deren Linearitaets-Annahme)
    # Speicher-Streaming (OOM-Fix 1.5B): Engram nach apply freigeben; Extraktion bleibt auf M0.
    import gc
    print(f"[{time.strftime('%H:%M:%S')}] Arm 1 (M0-Engrams, linear) ...", flush=True)
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
        print(f"  Arm1 nach {k}: {res['arm1_nll_steps'][-1]['nll']}", flush=True)

    # --- Arm 2: re-kalibriert (jeder Schnitt auf aktuellem Zustand)
    print(f"[{time.strftime('%H:%M:%S')}] Arm 2 (re-kalibriert) ...", flush=True)
    m2 = copy.deepcopy(m0)
    res["arm2_steps"] = []
    for i, k in enumerate(SEQ):
        layers = cut(m2, tokenizer, k)
        edited_layers |= layers
        remaining = [c for c in CONCEPTS if c not in SEQ[: i + 1]]
        drift = {c: round(cov_drift(concept_stats(m2, tokenizer, CONCEPTS[c]["forget"]), stats0[c]), 5)
                 for c in remaining}
        res["arm2_steps"].append({"cut": k, "nll": nll_all(m2, tokenizer), "cov_drift_remaining": drift})
        print(f"  Arm2 nach {k}: NLL={res['arm2_steps'][-1]['nll']}", flush=True)
        print(f"           Drift={drift}", flush=True)

    # --- V1: Arm1 vs Arm2
    d_scale = weight_dist(m0, m1, edited_layers)
    d_arms = weight_dist(m1, m2, edited_layers)
    res["V1_arm_divergence"] = {"dist_m0_m1": d_scale, "dist_m1_m2": d_arms,
                                "ratio": d_arms / (d_scale + 1e-12)}
    print(f"V1: dist(M1,M2)={d_arms:.6f}, Skala dist(M0,M1)={d_scale:.6f}, ratio={res['V1_arm_divergence']['ratio']:.3f}", flush=True)
    del m1

    # --- V2: Kommutativitaet (Arm-2-Stil), Ueberlapp-Paar vs fernes Paar
    print(f"[{time.strftime('%H:%M:%S')}] V2 Kommutativitaet ...", flush=True)
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
    out = f"results_seq_c4_{slug}.json" if MODEL_ID != "Qwen/Qwen3-0.6B" else "results_seq_c4.json"
    json.dump(res, open(os.path.join(OUT_DIR, out), "w"), indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] FERTIG -> {out}", flush=True)


if __name__ == "__main__":
    main()
