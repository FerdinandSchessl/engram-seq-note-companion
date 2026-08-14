#!/usr/bin/env python3
"""
Paper 2 Messblock 2 — Auswertung EXAKT nach PRE_REG_PAPER2.md Block-2-Nachtrag (08.07.):
Delta je Dialog gepaart (Zelle − Baseline) fuer var_sv_ai und E_ai_mean; Wilcoxon
zweiseitig ueber die 10 Dialog-Deltas; Median + r_rb + Bootstrap-CI (B=2000, seed=42).
P2-Kriterium: gleiches Vorzeichen Median-Delta ueber 3 Konzepte UND paarweise
ueberlappende CIs (alpha=0.6), separat je Metrik. alpha-Skalierung deskriptiv.
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon

DIR = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(DIR, "results_p2_trajectories.json")))
METRICS = ["var_sv_ai", "E_ai_mean", "epsilon_ai_mean"]
KEYS = ["E_insulin", "F_python", "A_eiffel"]
B, SEED = 2000, 42


def stats(deltas):
    x = np.asarray(deltas, dtype=float)
    x = x[~np.isnan(x)]
    nz = x[x != 0]
    if len(nz) == 0:
        return {"median": 0.0, "p": 1.0, "r_rb": 0.0, "ci": [0.0, 0.0], "n": len(x)}
    w = wilcoxon(x, alternative="two-sided")
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    wpos, wneg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    rng = np.random.default_rng(SEED)
    boots = np.median(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return {"median": float(np.median(x)), "p": float(w.pvalue),
            "r_rb": float((wpos - wneg) / (wpos + wneg)),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": len(x)}


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


base = d["baseline"]
res = {}
for c in d["cells"]:
    cell = {}
    for m in METRICS:
        deltas = [c["dialogs"][i][m] - base[i][m] for i in range(len(base))]
        cell[m] = stats(deltas)
        cell[m]["deltas"] = [round(float(x), 5) for x in deltas]
    res[f'{c["cut"]}_a{c["alpha"]}'] = cell

crit = {}
for m in ["var_sv_ai", "E_ai_mean"]:
    meds = [res[f"{k}_a0.6"][m]["median"] for k in KEYS]
    cis = [res[f"{k}_a0.6"][m]["ci"] for k in KEYS]
    same_sign = all(x > 0 for x in meds) or all(x < 0 for x in meds)
    pairwise = all(overlap(cis[i], cis[j]) for i in range(3) for j in range(i + 1, 3))
    crit[f"P2_{m}"] = {"medians": [round(x, 5) for x in meds], "same_sign": same_sign,
                       "cis_overlap": pairwise, "PASS": bool(same_sign and pairwise)}
    crit[f"alpha_scaling_{m}"] = {
        k: bool(abs(res[f"{k}_a0.6"][m]["median"]) > abs(res[f"{k}_a0.3"][m]["median"]))
        for k in KEYS}

out = {"analysis": "PRE_REG_PAPER2.md Block-2 frozen scheme", "B": B, "seed": SEED,
       "n_dialogs": len(base), "cells": res, "criteria": crit}
json.dump(out, open(os.path.join(DIR, "results_p2_traj_analysis.json"), "w"), indent=1)

L = ["# Messblock-2-Kurzreport (auto, Schema PRE_REG Block-2-Nachtrag)", ""]
L.append("| Zelle | Metrik | Median-Δ | p | r_rb | 95%-CI |")
L.append("|---|---|---|---|---|---|")
for cell, ms in res.items():
    for m in METRICS:
        s = ms[m]
        L.append(f"| {cell} | {m} | {s['median']:+.4f} | {s['p']:.4f} | "
                 f"{s['r_rb']:+.3f} | [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] |")
L += ["", "## Kriterien"]
for k, v in crit.items():
    L.append(f"- **{k}**: {json.dumps(v, ensure_ascii=False)}")
open(os.path.join(DIR, "P2_BLOCK2_REPORT.md"), "w").write("\n".join(L) + "\n")
print(json.dumps(crit, indent=1, ensure_ascii=False))
