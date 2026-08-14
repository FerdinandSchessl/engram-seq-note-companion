#!/usr/bin/env python3
"""
Entlastungs-Auswertung, Kriterien vor der Messung eingefroren. Primär Spearman(dNLL_real,
naehe) je Ziel + exakte Permutation (n=5 → 120 Perms, EINSEITIG in pre-registrierter
Richtung >0, Gegenrichtung mitberichtet). K1 real-vs-placebo × naehe. K2 Kontroll-Ziel.
"""
import json, os, itertools
import numpy as np
from scipy.stats import spearmanr

DIR = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(DIR, "results_entlastung.json")))


def perm_p(x, y, observed, side):
    """Exakte Permutations-p über alle Anordnungen von y. side: '+' (rho>=obs) / '-' (rho<=obs)."""
    xs = np.asarray(x)
    cnt = tot = 0
    for perm in itertools.permutations(range(len(y))):
        r = spearmanr(xs, np.asarray(y)[list(perm)]).statistic
        tot += 1
        if (side == "+" and r >= observed - 1e-12) or (side == "-" and r <= observed + 1e-12):
            cnt += 1
    return cnt / tot


out = {"analysis": "ZETTEL_entlastung_P3 frozen", "targets": {}}
for T, tdata in d["targets"].items():
    arms = tdata["arms"]
    naehe = [a["naehe_cos"] for a in arms]
    dreal = [a["dNLL_real"] for a in arms]
    dplac = [a["dNLL_placebo"] for a in arms]
    diff = [r - p for r, p in zip(dreal, dplac)]  # K1: real − placebo
    rho_real = spearmanr(dreal, naehe).statistic
    rho_k1 = spearmanr(diff, naehe).statistic
    res = {
        "cut_effective": tdata["cut_effective"], "nll_solo": tdata["nll_solo"],
        "arms": [{"X": a["X"], "naehe": a["naehe_cos"], "dNLL_real": a["dNLL_real"],
                  "dNLL_placebo": a["dNLL_placebo"], "real_minus_placebo": round(r - p, 4),
                  "dose": a["dose_mean_rel_frob"]}
                 for a, r, p in zip(arms, dreal, dplac)],
        "primary_spearman_dNLLreal_vs_naehe": round(rho_real, 4),
        "perm_p_preregistered_gt0": round(perm_p(dreal, naehe, rho_real, "+"), 4),
        "perm_p_opposite_lt0": round(perm_p(dreal, naehe, rho_real, "-"), 4),
        "K1_spearman_realMinusPlacebo_vs_naehe": round(rho_k1, 4),
        "K1_mean_real_minus_placebo": round(float(np.mean(diff)), 4),
    }
    out["targets"][T] = res

json.dump(out, open(os.path.join(DIR, "results_entlastung_analysis.json"), "w"), indent=1)

L = ["# Entlastungs-Test — Report (Schema ZETTEL_entlastung_P3)", ""]
for T, r in out["targets"].items():
    L.append(f"## Ziel {T}  (cut_effective={r['cut_effective']:+.3f}, nll_solo={r['nll_solo']})")
    L.append("| X (Zweitschnitt) | Nähe cos | ΔNLL real | ΔNLL placebo | real−placebo | Dosis |")
    L.append("|---|---|---|---|---|---|")
    for a in r["arms"]:
        L.append(f"| {a['X']} | {a['naehe']:+.3f} | {a['dNLL_real']:+.3f} | "
                 f"{a['dNLL_placebo']:+.3f} | {a['real_minus_placebo']:+.3f} | {a['dose']:.4f} |")
    L.append(f"\n- **Primär** Spearman(ΔNLL_real, Nähe) = **{r['primary_spearman_dNLLreal_vs_naehe']:+.3f}** "
             f"(Perm-p pre-reg >0: {r['perm_p_preregistered_gt0']}; Gegenrichtung <0: {r['perm_p_opposite_lt0']})")
    L.append(f"- **K1** Spearman(real−placebo, Nähe) = {r['K1_spearman_realMinusPlacebo_vs_naehe']:+.3f}; "
             f"mean(real−placebo) = {r['K1_mean_real_minus_placebo']:+.3f}\n")
open(os.path.join(DIR, "ENTLASTUNG_REPORT.md"), "w").write("\n".join(L) + "\n")
print(json.dumps(out, indent=1, ensure_ascii=False))
