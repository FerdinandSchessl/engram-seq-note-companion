#!/usr/bin/env python3
"""
Paper 2 Messblock 1 — Auswertung EXAKT nach PRE_REG_PAPER2.md (eingefroren 2026-07-07):
Wilcoxon signed-rank (zweiseitig) ueber gepaarte per-Prompt-Deltas, Median-Delta,
rang-biseriale r, Bootstrap-95%-CI des Medians (B=2000, seed=42 je Marker-Zelle).
Kriterien: P1b (Vorzeichen gleich + CIs ueberlappen, alpha=0.6 Instruct),
alpha-Skalierung (deskriptiv), P3 (|med|_Base < |med|_Instruct in >=2/3 Konzepten).
"""
import json, os, argparse
import numpy as np
from scipy.stats import wilcoxon

DIR = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--instruct", default="results_p2_markers_Qwen2-0_5B-Instruct.json")
_ap.add_argument("--base", default="results_p2_markers_Qwen2-0_5B.json")
_ap.add_argument("--suffix", default="", help="Suffix fuer Output-Dateien, z.B. _qwen3")
_args = _ap.parse_args()
FILES = {"instruct": _args.instruct, "base": _args.base}
SUF = _args.suffix
MARKERS = ["refusal", "hedge", "syco_low", "syco_high"]
P1B = ["hedge", "syco_low", "syco_high"]
KEYS = ["E_insulin", "F_python", "A_eiffel"]
B, SEED = 2000, 42


def stats(deltas):
    d = np.asarray(deltas, dtype=float)
    nz = d[d != 0]
    if len(nz) == 0:
        return {"median": 0.0, "p": 1.0, "r_rb": 0.0, "ci": [0.0, 0.0], "n": len(d)}
    w = wilcoxon(d, alternative="two-sided")
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    wpos, wneg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    rng = np.random.default_rng(SEED)
    boots = np.median(rng.choice(d, size=(B, len(d)), replace=True), axis=1)
    return {"median": float(np.median(d)), "p": float(w.pvalue),
            "r_rb": float((wpos - wneg) / (wpos + wneg)),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": len(d)}


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


res = {}
for tag, fn in FILES.items():
    data = json.load(open(os.path.join(DIR, fn)))
    base = data["baseline"]["markers"]
    cells = {}
    for c in data["cells"]:
        cell = {}
        for m in MARKERS:
            deltas = [x - y for x, y in zip(c["markers"][m], base[m])]
            s = stats(deltas)
            if m in ("refusal", "hedge"):
                s["neutral"] = stats(deltas[:30])
                s["borderline"] = stats(deltas[30:])
            cell[m] = s
        cell["dnll_target"] = round(c["nll"][c["cut"]] - data["baseline"]["nll"][c["cut"]], 4)
        cells[f'{c["cut"]}_a{c["alpha"]}'] = cell
    res[tag] = cells

# --- Kriterien (fixiert) ---
crit = {}
for m in P1B:
    meds = [res["instruct"][f"{k}_a0.6"][m]["median"] for k in KEYS]
    cis = [res["instruct"][f"{k}_a0.6"][m]["ci"] for k in KEYS]
    same_sign = all(x > 0 for x in meds) or all(x < 0 for x in meds)
    pairwise = all(overlap(cis[i], cis[j]) for i in range(3) for j in range(i + 1, 3))
    crit[f"P1b_{m}"] = {"medians": [round(x, 5) for x in meds], "same_sign": same_sign,
                        "cis_overlap": pairwise, "PASS": bool(same_sign and pairwise)}

for m in P1B + ["refusal"]:
    crit[f"alpha_scaling_{m}"] = {
        k: bool(abs(res["instruct"][f"{k}_a0.6"][m]["median"])
                > abs(res["instruct"][f"{k}_a0.3"][m]["median"])) for k in KEYS}

for m in P1B:
    wins = [k for k in KEYS
            if abs(res["base"][f"{k}_a0.6"][m]["median"])
            < abs(res["instruct"][f"{k}_a0.6"][m]["median"])]
    crit[f"P3_{m}"] = {"instruct_groesser_in": wins, "PASS": len(wins) >= 2}

crit["syco_grading_instruct_a0.6"] = {
    k: bool(abs(res["instruct"][f"{k}_a0.6"]["syco_high"]["median"])
            >= abs(res["instruct"][f"{k}_a0.6"]["syco_low"]["median"])) for k in KEYS}

out = {"analysis": "PRE_REG_PAPER2.md frozen scheme", "B": B, "seed": SEED,
       "cells": res, "criteria": crit}
json.dump(out, open(os.path.join(DIR, f"results_p2_analysis{SUF}.json"), "w"), indent=1)

# --- Kompakt-Report ---
L = ["# Messblock-1-Kurzreport (auto, Schema PRE_REG_PAPER2.md)", ""]
for tag in ("instruct", "base"):
    L.append(f"## {tag}")
    L.append("| Zelle | Marker | Median-Δ | p | r_rb | 95%-CI |")
    L.append("|---|---|---|---|---|---|")
    for cell, ms in res[tag].items():
        for m in MARKERS:
            s = ms[m]
            L.append(f"| {cell} | {m} | {s['median']:+.4f} | {s['p']:.4f} | "
                     f"{s['r_rb']:+.3f} | [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] |")
    L.append("")
L.append("## Kriterien")
for k, v in crit.items():
    L.append(f"- **{k}**: {json.dumps(v, ensure_ascii=False)}")
open(os.path.join(DIR, f"P2_BLOCK1_REPORT{SUF}.md"), "w").write("\n".join(L) + "\n")
print(json.dumps(crit, indent=1, ensure_ascii=False))
