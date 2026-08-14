#!/usr/bin/env python3
"""Robustheits-Rechnungen Note 2 (10.07.2026, report-only, NICHT konfirmatorisch).

Rechnet aus den VORHANDENEN Roh-JSONs (kein Modell-Lauf):
  1) Qwen3-Laengen-Kontrolle aus gespeicherten Transkripten (ai_texts):
     Med-dLen + Med-dE_ai je Zelle/Arm (Laengen-Ritterschnitt auf der Replikations-Charge).
  2) Vorzeichen-Zaehlungen dE_ai je Dialog (alpha0.6, beide Chargen, engram vs placebo).
  3) Sensor-Capture-Check: absolute Antwortlaengen je Zelle (degenerierte Prueflkoerper).
  4) Schwere-Schwaenze-Seitigkeit: max|dE_ai| je Zelle engram vs placebo.

Bestimmt fuer: ai-engram-seq Repo-Root (neben den results_*.json), danach in
verify_numbers_note2.py referenzierte Zahlen aufnehmen. Output:
results_p2_robustness.json + P2_ROBUSTNESS_REPORT.md
"""
import json
import os

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(DIR, f)))
tr = J("results_p2_trajectories.json")       # Qwen2 Engram (+Baseline)
pl = J("results_p2_traj_placebo.json")       # Qwen2 Placebo a0.6
tq = J("results_p2_traj_qwen3.json")         # Qwen3 kombiniert (engram+placebo, arm-Feld)

mlen = lambda d: float(np.mean([len(t) for t in d["ai_texts"]]))
med = lambda x: float(np.median(x))

out = {"note": "report-only Robustheits-Rechnungen 10.07.2026, PRE_REG-Kriterien unangetastet",
       "qwen3_len": {}, "sign_counts": {}, "sensor_capture": {}, "tails": {}}

# --- 1) Qwen3 Laengen aus Transkripten ---
arms = {(c["cut"], c["arm"]): c for c in tq["cells"]}
for (k, arm), c in sorted(arms.items()):
    dl = [mlen(c["dialogs"][i]) - mlen(tq["baseline"][i]) for i in range(len(tq["baseline"]))]
    de = [c["dialogs"][i]["E_ai_mean"] - tq["baseline"][i]["E_ai_mean"] for i in range(len(tq["baseline"]))]
    out["qwen3_len"][f"{k}_{arm}"] = {"med_dlen": round(med(dl), 1), "med_dE_ai": round(med(de), 2),
                                      "n_pos_dE_ai": sum(1 for x in de if x > 0), "n": len(de)}

# --- 2+4) Vorzeichen + Tails, Qwen2 a0.6 + Qwen3 ---
eng2 = {c["cut"]: c for c in tr["cells"] if c["alpha"] == 0.6}
pla2 = {c["cut"]: c for c in pl["cells"]}
def signs(cell, base):
    de = [cell["dialogs"][i]["E_ai_mean"] - base[i]["E_ai_mean"] for i in range(len(base))]
    return {"med_dE_ai": round(med(de), 2), "n_pos": sum(1 for x in de if x > 0),
            "n": len(de), "max_abs_dE_ai": round(max(de, key=abs), 1)}
for k in ("E_insulin", "F_python", "A_eiffel"):
    out["sign_counts"][f"Q2_{k}_engram"] = signs(eng2[k], tr["baseline"])
    out["sign_counts"][f"Q2_{k}_placebo"] = signs(pla2[k], tr["baseline"])
    out["sign_counts"][f"Q3_{k}_engram"] = signs(arms[(k, "engram")], tq["baseline"])
    out["sign_counts"][f"Q3_{k}_placebo"] = signs(arms[(k, "placebo")], tq["baseline"])

# --- 3) Sensor-Capture: absolute Laengen ---
for tag, base, cells in (("Q2", tr["baseline"], [(k, a, (eng2 if a == "engram" else pla2)[k])
                                                 for k in eng2 for a in ("engram", "placebo")]),
                         ("Q3", tq["baseline"], [(k, a, arms[(k, a)]) for (k, a) in arms])):
    out["sensor_capture"][f"{tag}_baseline_mean_len"] = round(float(np.mean([mlen(d) for d in base])), 0)
    for k, a, c in cells:
        ls = [mlen(d) for d in c["dialogs"]]
        out["sensor_capture"][f"{tag}_{k}_{a}"] = {"mean_len": round(float(np.mean(ls)), 0),
                                                   "min_dialog_mean_len": round(min(ls), 0)}

json.dump(out, open(os.path.join(DIR, "results_p2_robustness.json"), "w"), indent=1)

L = ["# Robustheits-Rechnungen Note 2 (10.07., report-only, nicht konfirmatorisch)", "",
     "PRE_REG-Kriterien unangetastet; Rechnungen ausschliesslich aus vorhandenen Roh-JSONs.", "",
     "## Qwen3 Laengen-Kontrolle (aus gespeicherten Transkripten)",
     "| Zelle | Arm | Med-dLen | Med-dE_ai | dE_ai>0 |", "|---|---|---|---|---|"]
for key, v in out["qwen3_len"].items():
    k, arm = key.rsplit("_", 1)
    L.append(f"| {k} | {arm} | {v['med_dlen']:+.1f} | {v['med_dE_ai']:+.2f} | {v['n_pos_dE_ai']}/{v['n']} |")
L += ["", "## Vorzeichen-Zaehlungen + Tails (dE_ai je Dialog)",
      "| Zelle | Med-dE_ai | >0 | max|dE_ai| |", "|---|---|---|---|"]
for key, v in out["sign_counts"].items():
    L.append(f"| {key} | {v['med_dE_ai']:+.2f} | {v['n_pos']}/{v['n']} | {v['max_abs_dE_ai']:+.1f} |")
L += ["", "## Sensor-Capture (absolute Antwortlaengen, Zeichen)", "```",
      json.dumps(out["sensor_capture"], indent=1), "```"]
open(os.path.join(DIR, "P2_ROBUSTNESS_REPORT.md"), "w").write("\n".join(L) + "\n")
print(json.dumps(out["qwen3_len"], indent=1))
print({k: (v["n_pos"], v["n"]) for k, v in out["sign_counts"].items()})
