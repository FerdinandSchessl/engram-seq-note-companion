#!/usr/bin/env python3
"""
Paper 2 — P4-Auswertung EXAKT nach PRE_REG_PAPER2.md P4-Nachtrag (09.07.):
je Zelle/Arm Delta-Stats (Wilcoxon, Median, r_rb, Bootstrap-CI B=2000 seed=42),
direkter Paarvergleich Delta_Engram vs Delta_Placebo (Wilcoxon ueber per-Prompt-
Differenzen), Fairness-Gate Retain-dNLL_Placebo in [0.5x, 2x] Engram,
P4a: |Med-dhedge_Placebo| < 0.5 x |Med-dhedge_Engram| in >=2/3 Konzepten (a=0.6, Instruct).
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon

DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {"instruct": "results_p2_placebo_Qwen2-0_5B-Instruct.json",
         "base": "results_p2_placebo_Qwen2-0_5B.json"}
MARKERS = ["refusal", "hedge", "syco_low", "syco_high"]
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


res = {}
for tag, fn in FILES.items():
    data = json.load(open(os.path.join(DIR, fn)))
    base = data["baseline"]
    cells = {}
    for c in data["cells"]:
        cell = {"dose_mean_rel_frob": c["dose_mean_rel_frob"],
                "bias_energy_share": c.get("bias_energy_share"),
                "n_layers": c["n_layers"], "placebo_seed": c["placebo_seed"]}
        for arm in ("engram", "placebo"):
            a = {}
            for m in MARKERS:
                deltas = [x - y for x, y in zip(c[arm]["markers"][m], base["markers"][m])]
                a[m] = stats(deltas)
            a["d_retain_nll"] = round(c[arm]["retain_nll"] - base["retain_nll"], 4)
            a["d_target_nll"] = round(c[arm]["nll"][c["cut"]] - base["nll"][c["cut"]], 4)
            a["d_answer_len"] = round(c[arm]["mean_answer_len"] - base["mean_answer_len"], 1)
            cell[arm] = a
        for m in MARKERS:
            de = [x - y for x, y in zip(c["engram"]["markers"][m], base["markers"][m])]
            dp = [x - y for x, y in zip(c["placebo"]["markers"][m], base["markers"][m])]
            cell[f"pair_{m}"] = stats([a - b for a, b in zip(de, dp)])
        ratio = (cell["placebo"]["d_retain_nll"] / cell["engram"]["d_retain_nll"]
                 if cell["engram"]["d_retain_nll"] else float("inf"))
        cell["fairness_ratio"] = round(ratio, 3)
        cell["fairness_ok"] = bool(0.5 <= ratio <= 2.0)
        cells[f'{c["cut"]}_a{c["alpha"]}'] = cell
    res[tag] = cells

# --- P4a (fixiert): hedge, a=0.6, Instruct ---
crit = {}
rows = []
for k in KEYS:
    cell = res["instruct"][f"{k}_a0.6"]
    e, p = abs(cell["engram"]["hedge"]["median"]), abs(cell["placebo"]["hedge"]["median"])
    rows.append({"concept": k, "abs_med_engram": round(e, 4), "abs_med_placebo": round(p, 4),
                 "ratio_p_over_e": round(p / (e + 1e-12), 3), "edit_specific": bool(p < 0.5 * e),
                 "fairness_ok": cell["fairness_ok"], "fairness_ratio": cell["fairness_ratio"]})
n_spec = sum(r["edit_specific"] for r in rows)
n_spec_fair = sum(r["edit_specific"] for r in rows if r["fairness_ok"])
crit["P4a_hedge"] = {"cells": rows, "edit_specific_count": n_spec,
                     "PASS_alle_zellen": bool(n_spec >= 2),
                     "gewertete_zellen_fair": [r["concept"] for r in rows if r["fairness_ok"]],
                     "PASS_nur_faire_zellen": bool(n_spec_fair >= 2)}

# Analog (Bericht, kein Gate): answer_len + syco + refusal, a=0.6 Instruct
for name, getter in [("answer_len", lambda c, arm: abs(c[arm]["d_answer_len"]))]:
    rows2 = []
    for k in KEYS:
        cell = res["instruct"][f"{k}_a0.6"]
        e, p = getter(cell, "engram"), getter(cell, "placebo")
        rows2.append({"concept": k, "abs_engram": e, "abs_placebo": p,
                      "ratio": round(p / (e + 1e-12), 3)})
    crit[f"report_{name}"] = rows2

out = {"analysis": "PRE_REG_PAPER2.md P4 frozen scheme", "B": B, "seed": SEED,
       "cells": res, "criteria": crit}
json.dump(out, open(os.path.join(DIR, "results_p2_placebo_analysis.json"), "w"), indent=1)

L = ["# P4-Placebo-Kurzreport (auto, Schema PRE_REG P4-Nachtrag)", ""]
for tag in ("instruct", "base"):
    L.append(f"## {tag}")
    L.append("| Zelle | fair? (P/E-Retain) | Marker | Med-Δ Engram | Med-Δ Placebo | Paar-p | Δlen E/P |")
    L.append("|---|---|---|---|---|---|---|")
    for cname, cell in res[tag].items():
        for m in MARKERS:
            L.append(f"| {cname} | {'JA' if cell['fairness_ok'] else 'NEIN'} "
                     f"({cell['fairness_ratio']}) | {m} | "
                     f"{cell['engram'][m]['median']:+.4f} | {cell['placebo'][m]['median']:+.4f} | "
                     f"{cell[f'pair_{m}']['p']:.4f} | "
                     f"{cell['engram']['d_answer_len']:+.0f}/{cell['placebo']['d_answer_len']:+.0f} |")
    L.append("")
L.append("## Kriterien")
L.append(json.dumps(crit, indent=1, ensure_ascii=False))
open(os.path.join(DIR, "P2_P4_REPORT.md"), "w").write("\n".join(L) + "\n")
print(json.dumps(crit, indent=1, ensure_ascii=False))
