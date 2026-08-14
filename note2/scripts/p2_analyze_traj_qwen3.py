#!/usr/bin/env python3
"""P4b-Qwen3-Auswertung: Engram vs Placebo je Metrik/Konzept (gepaart je Dialog gegen
Baseline), Kriterium symmetrisch zu P4b-Qwen2: edit-spezifisch = |med Placebo| < 0.5|med Engram|
in >=2/3. Stats: Wilcoxon/Median/r_rb/Bootstrap (B=2000, seed=42)."""
import json, os
import numpy as np
from scipy.stats import wilcoxon

DIR = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(DIR, "results_p2_traj_qwen3.json")))
base = d["baseline"]
METRICS = ["var_sv_ai", "E_ai_mean"]
KEYS = ["E_insulin", "F_python", "A_eiffel"]
B, SEED = 2000, 42


def stats(deltas):
    x = np.asarray(deltas, float); x = x[~np.isnan(x)]
    nz = x[x != 0]
    if len(nz) == 0:
        return {"median": 0.0, "p": 1.0, "r_rb": 0.0, "ci": [0.0, 0.0]}
    w = wilcoxon(x, alternative="two-sided")
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    wpos, wneg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    rng = np.random.default_rng(SEED)
    bo = np.median(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return {"median": float(np.median(x)), "p": float(w.pvalue),
            "r_rb": float((wpos - wneg) / (wpos + wneg)),
            "ci": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]}


cell = {(c["cut"], c["arm"]): c["dialogs"] for c in d["cells"]}
res, crit = {}, {}
for m in METRICS:
    rows = []
    for k in KEYS:
        de = [cell[(k, "engram")][i][m] - base[i][m] for i in range(len(base))]
        dp = [cell[(k, "placebo")][i][m] - base[i][m] for i in range(len(base))]
        se, sp = stats(de), stats(dp)
        res[f"{k}_{m}"] = {"engram": se, "placebo": sp}
        e, p = abs(se["median"]), abs(sp["median"])
        rows.append({"concept": k, "med_engram": round(se["median"], 4),
                     "p_engram": round(se["p"], 4), "med_placebo": round(sp["median"], 4),
                     "p_placebo": round(sp["p"], 4), "ratio": round(p / (e + 1e-12), 3),
                     "edit_specific": bool(p < 0.5 * e)})
    crit[m] = {"cells": rows, "edit_specific_count": sum(r["edit_specific"] for r in rows),
               "PASS": bool(sum(r["edit_specific"] for r in rows) >= 2)}

json.dump({"model": d["model"], "cells": res, "criteria": crit},
          open(os.path.join(DIR, "results_p2_traj_qwen3_analysis.json"), "w"), indent=1)
L = ["# P4b-Qwen3-Report (Trajektorien-Placebo, alpha=0.6)", ""]
for m in METRICS:
    L.append(f"## {m}")
    L.append("| Konzept | med Engram (p) | med Placebo (p) | Ratio P/E | edit-spezifisch? |")
    L.append("|---|---|---|---|---|")
    for r in crit[m]["cells"]:
        L.append(f"| {r['concept']} | {r['med_engram']:+.3f} ({r['p_engram']}) | "
                 f"{r['med_placebo']:+.3f} ({r['p_placebo']}) | {r['ratio']} | "
                 f"{'JA' if r['edit_specific'] else 'nein'} |")
    L.append(f"\n→ {crit[m]['edit_specific_count']}/3 edit-spezifisch → "
             f"**{'EDIT-SPEZIFISCH (repliziert)' if crit[m]['PASS'] else 'nicht bestaetigt'}**\n")
open(os.path.join(DIR, "P2_P4b_QWEN3_REPORT.md"), "w").write("\n".join(L) + "\n")
print(json.dumps(crit, indent=1, ensure_ascii=False))
