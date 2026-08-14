#!/usr/bin/env python3
"""
Paper 2 Messblock 2 — deklarierte Sensitivitaetsanalyse "Laengen-Kontrolle Block 2".
Prueft, ob die Block-2-Effekte (var_sv_ai, E_ai_mean) mit der mittleren Antwort-
laenge (Zeichen) kovariieren bzw. nach Laengen-Kontrolle richtungsstabil bleiben.
Stats-Schema EXAKT uebernommen aus p2_analyze_traj.py: stats() (Wilcoxon zweiseitig,
Median, rang-biseriale r_rb, Bootstrap-CI B=2000 seed=42).

(i)   Je Zelle: Delta-len_i = len_i(Zelle) - len_i(Baseline) je Dialog i, stats().
(ii)  Je Zelle: Spearman(Delta-metrik, Delta-len) ueber die 10 Dialoge (Wert + p).
(iii) Gepoolt (6 Zellen = 60 Paare): Spearman(Delta-metrik, Delta-len) je Metrik;
      OLS Delta-metrik = a + b*Delta-len (numpy.polyfit); je Zelle Median der
      Residuen; Richtungsstabilitaet = Vorzeichen(Residuen-Median) == Vorzeichen
      (rohes Median-Delta) in >=5/6 Zellen -> PASS.
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon, spearmanr

DIR = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(DIR, "results_p2_trajectories.json")))
METRICS = ["var_sv_ai", "E_ai_mean"]
B, SEED = 2000, 42


def stats(deltas):
    """EXAKT aus p2_analyze_traj.py uebernommen."""
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


def mean_len(dialog):
    """Mittlere Antwortlaenge (Zeichen) ueber die ai_texts eines Dialogs."""
    return float(np.mean([len(t) for t in dialog["ai_texts"]]))


def same_sign(a, b):
    """Gleiches (echtes) Vorzeichen; 0 zaehlt nicht als Match."""
    return (a > 0 and b > 0) or (a < 0 and b < 0)


base = d["baseline"]
n = len(base)
base_len = [mean_len(b) for b in base]

cell_labels = []
cell_dlen = {}                              # label -> [10] Delta-len
cell_dmetric = {m: {} for m in METRICS}     # metrik -> label -> [10] Delta-metrik
for c in d["cells"]:
    label = f'{c["cut"]}_a{c["alpha"]}'
    cell_labels.append(label)
    cell_dlen[label] = [mean_len(c["dialogs"][i]) - base_len[i] for i in range(n)]
    for m in METRICS:
        cell_dmetric[m][label] = [c["dialogs"][i][m] - base[i][m] for i in range(n)]

# --- (i) Delta-len je Zelle + (ii) Spearman je Zelle ---
cells_out = {}
for label in cell_labels:
    dl = cell_dlen[label]
    entry = {"dlen": stats(dl)}
    entry["dlen"]["deltas"] = [round(x, 4) for x in dl]
    sp = {}
    for m in METRICS:
        rho, p = spearmanr(cell_dmetric[m][label], dl)
        sp[m] = {"rho": float(rho), "p": float(p)}
    entry["spearman"] = sp
    cells_out[label] = entry

# --- (iii) gepoolt (60 Paare): Spearman + OLS-Residuen + Richtungsstabilitaet ---
pooled_dlen = np.concatenate([cell_dlen[l] for l in cell_labels])
pooled = {"n_pairs": int(len(pooled_dlen)), "spearman": {}, "ols": {}}
direction = {}
for m in METRICS:
    pooled_dm = np.concatenate([cell_dmetric[m][l] for l in cell_labels])
    rho, p = spearmanr(pooled_dm, pooled_dlen)
    pooled["spearman"][m] = {"rho": float(rho), "p": float(p)}
    coeffs = np.polyfit(pooled_dlen, pooled_dm, 1)  # [slope b, intercept a]
    pooled["ols"][m] = {"slope": float(coeffs[0]), "intercept": float(coeffs[1])}
    per_cell = {}
    n_match = 0
    for label in cell_labels:
        dl = np.asarray(cell_dlen[label], dtype=float)
        dm = np.asarray(cell_dmetric[m][label], dtype=float)
        resid = dm - np.polyval(coeffs, dl)          # gepoolter OLS-Fit, zellweise Residuen
        res_med = float(np.median(resid))
        raw_med = float(np.median(dm))
        match = same_sign(res_med, raw_med)
        n_match += int(match)
        per_cell[label] = {"raw_median_delta": raw_med, "residual_median": res_med,
                           "raw_sign": int(np.sign(raw_med)),
                           "resid_sign": int(np.sign(res_med)), "match": bool(match)}
        cells_out[label].setdefault("raw_median_delta", {})[m] = raw_med
        cells_out[label].setdefault("residual_median", {})[m] = res_med
        cells_out[label].setdefault("sign_match", {})[m] = bool(match)
    direction[m] = {"n_match": int(n_match), "n_cells": len(cell_labels),
                    "PASS": bool(n_match >= 5), "per_cell": per_cell}

out = {"analysis": "P2 Block-2 Sensitivitaet Laengen-Kontrolle",
       "stats_schema": "p2_analyze_traj.py stats() (Wilcoxon two-sided, median, r_rb, bootstrap CI B=2000 seed=42)",
       "B": B, "seed": SEED, "n_dialogs": n, "metrics": METRICS,
       "cells": cells_out, "pooled": pooled, "direction_stability": direction}
json.dump(out, open(os.path.join(DIR, "results_p2_len_control.json"), "w"), indent=1)

# --- Markdown-Report ---
L = ["# P2 Block-2 — Laengen-Kontrolle (Sensitivitaetsanalyse)", ""]
L.append(f"Stats-Schema: {out['stats_schema']}. n_dialogs={n}. Laenge = Zeichen "
         f"(mean(len(t) fuer t in ai_texts)).")
L += ["", "## (i) Delta-len je Zelle (Zelle - Baseline, 10 gepaarte Dialoge)", "",
      "| Zelle | Median-Δlen | p | r_rb | 95%-CI |", "|---|---|---|---|---|"]
for label in cell_labels:
    s = cells_out[label]["dlen"]
    L.append(f"| {label} | {s['median']:+.2f} | {s['p']:.4f} | {s['r_rb']:+.3f} | "
             f"[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] |")

L += ["", "## (ii) Spearman(Δmetrik, Δlen) je Zelle (n=10)", "",
      "| Zelle | ρ(Δvar_sv_ai, Δlen) | p | ρ(ΔE_ai_mean, Δlen) | p |",
      "|---|---|---|---|---|"]
for label in cell_labels:
    sv = cells_out[label]["spearman"]["var_sv_ai"]
    ea = cells_out[label]["spearman"]["E_ai_mean"]
    L.append(f"| {label} | {sv['rho']:+.3f} | {sv['p']:.4f} | {ea['rho']:+.3f} | {ea['p']:.4f} |")

L += ["", "## (iii) Gepoolt (6 Zellen = 60 Paare)", "",
      "| Metrik | ρ(Δmetrik, Δlen) | p | OLS slope b | intercept a |",
      "|---|---|---|---|---|"]
for m in METRICS:
    sp = pooled["spearman"][m]; ols = pooled["ols"][m]
    L.append(f"| {m} | {sp['rho']:+.3f} | {sp['p']:.4f} | {ols['slope']:+.6g} | {ols['intercept']:+.6g} |")

L += ["", "### Richtungsstabilitaet nach Laengen-Kontrolle (Residuen der gepoolten OLS)"]
for m in METRICS:
    ds = direction[m]
    verdict = "PASS" if ds["PASS"] else "FAIL"
    L += ["", f"**{m}: {verdict}** ({ds['n_match']}/{ds['n_cells']} Zellen gleiches "
          f"Vorzeichen; PASS = >=5/6)", "",
          "| Zelle | rohes Median-Δ | Vorz. | Residuen-Median | Vorz. | Match |",
          "|---|---|---|---|---|---|"]
    for label in cell_labels:
        pc = ds["per_cell"][label]
        L.append(f"| {label} | {pc['raw_median_delta']:+.5f} | {pc['raw_sign']:+d} | "
                 f"{pc['residual_median']:+.5f} | {pc['resid_sign']:+d} | "
                 f"{'match' if pc['match'] else 'NO'} |")

open(os.path.join(DIR, "P2_LEN_CONTROL_REPORT.md"), "w").write("\n".join(L) + "\n")

print("=== DIRECTION STABILITY ===")
for m in METRICS:
    ds = direction[m]
    print(f"{m}: {'PASS' if ds['PASS'] else 'FAIL'}  n_match={ds['n_match']}/{ds['n_cells']}")
    for label in cell_labels:
        pc = ds["per_cell"][label]
        print(f"   {label:16s} raw_med={pc['raw_median_delta']:+.5f} (sign {pc['raw_sign']:+d})"
              f"  resid_med={pc['residual_median']:+.5f} (sign {pc['resid_sign']:+d})"
              f"  match={pc['match']}")
print("\n=== DELTA-LEN MEDIANS (per cell) ===")
for label in cell_labels:
    s = cells_out[label]["dlen"]
    print(f"   {label:16s} median_dlen={s['median']:+.2f}  p={s['p']:.4f}")
print("\n=== POOLED SPEARMAN (60 pairs) ===")
for m in METRICS:
    sp = pooled["spearman"][m]
    print(f"   {m:12s} rho={sp['rho']:+.4f}  p={sp['p']:.4f}")
print("\nWROTE: results_p2_len_control.json , P2_LEN_CONTROL_REPORT.md")
