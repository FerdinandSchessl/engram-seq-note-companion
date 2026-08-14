#!/usr/bin/env python3
"""
P4b-Auswertung nach PRE_REG_PAPER2.md P4b-Nachtrag: Placebo-Trajektorien vs
Engram-alpha0.6, Baseline aus results_p2_trajectories.json (wiederverwendet).
Kriterium symmetrisch zu P4a: edit-spezifisch, wenn |Med-D_Placebo| < 0.5x|Med-D_Engram|
in >=2/3 Konzepten je Metrik. Stats-Schema identisch (Wilcoxon/Median/r_rb/Bootstrap).
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon

DIR = os.path.dirname(os.path.abspath(__file__))
traj = json.load(open(os.path.join(DIR, "results_p2_trajectories.json")))
plac = json.load(open(os.path.join(DIR, "results_p2_traj_placebo.json")))
base = traj["baseline"]
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


eng_cell = {c["cut"]: c for c in traj["cells"] if c["alpha"] == 0.6}
pla_cell = {c["cut"]: c for c in plac["cells"]}
res, crit = {}, {}
for m in METRICS:
    rows = []
    for k in KEYS:
        de = [eng_cell[k]["dialogs"][i][m] - base[i][m] for i in range(len(base))]
        dp = [pla_cell[k]["dialogs"][i][m] - base[i][m] for i in range(len(base))]
        se, sp = stats(de), stats(dp)
        res[f"{k}_{m}"] = {"engram": se, "placebo": sp}
        e, p = abs(se["median"]), abs(sp["median"])
        rows.append({"concept": k, "med_engram": round(se["median"], 4),
                     "med_placebo": round(sp["median"], 4),
                     "ratio_p_over_e": round(p / (e + 1e-12), 3),
                     "edit_specific": bool(p < 0.5 * e)})
    crit[m] = {"cells": rows, "edit_specific_count": sum(r["edit_specific"] for r in rows),
               "PASS_edit_specific": bool(sum(r["edit_specific"] for r in rows) >= 2)}

out = {"analysis": "PRE_REG P4b frozen scheme", "cells": res, "criteria": crit}
json.dump(out, open(os.path.join(DIR, "results_p2_traj_placebo_analysis.json"), "w"), indent=1)

L = ["# P4b-Kurzreport (Trajektorien-Placebo vs Engram alpha0.6)", ""]
for m in METRICS:
    L.append(f"## {m}")
    L.append("| Konzept | Med-Δ Engram | Med-Δ Placebo | Ratio P/E | edit-spezifisch? |")
    L.append("|---|---|---|---|---|")
    for r in crit[m]["cells"]:
        L.append(f"| {r['concept']} | {r['med_engram']:+.4f} | {r['med_placebo']:+.4f} | "
                 f"{r['ratio_p_over_e']} | {'JA' if r['edit_specific'] else 'nein'} |")
    L.append(f"\n→ edit-spezifisch in {crit[m]['edit_specific_count']}/3 → "
             f"**{'edit-spezifisch' if crit[m]['PASS_edit_specific'] else 'DOSIS-GENERISCH (Null-Lesart)'}**\n")
L.append("bias_energy_share Placebo-Zellen: " +
         str({c["cut"]: c["bias_energy_share"] for c in plac["cells"]}))
open(os.path.join(DIR, "P2_P4b_REPORT.md"), "w").write("\n".join(L) + "\n")
print(json.dumps(crit, indent=1, ensure_ascii=False))
