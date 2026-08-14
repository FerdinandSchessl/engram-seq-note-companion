#!/usr/bin/env python3
"""Verifiziert jede in NOTE2_DRAFT_surgical.md zitierte Kernzahl gegen die rohen
Auswertungs-JSONs. stdlib only. Erwartete Ausgabe: alle PASS."""
import json, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
J = lambda f: json.load(open(os.path.join(D, f)))
q2 = J("results_p2_analysis.json")            # Qwen2 Marker
q3 = J("results_p2_analysis_qwen3.json")       # Qwen3 Marker
pl = J("results_p2_placebo_analysis.json")     # P4a Placebo (Marker)
tp = J("results_p2_traj_placebo_analysis.json")  # P4b Qwen2 (Trajektorien)
tq = J("results_p2_traj_qwen3_analysis.json")  # P4b Qwen3 (Trajektorien)

checks = []
def ck(name, actual, expected, tol=5e-3):
    checks.append((name, actual, expected, abs(actual - expected) <= tol))
def ck_lt(name, actual, bound):
    checks.append((name, actual, f"<{bound}", abs(actual) < bound))

# --- §4 P1a Refusal lokalisiert (Qwen2 instruct a0.6) ---
rf2 = lambda k: q2["cells"]["instruct"][f"{k}_a0.6"]["refusal"]["median"]
ck("§4 refusal Qwen2 A_eiffel -0.95", rf2("A_eiffel"), -0.9502)
ck("§4 refusal Qwen2 F_python +0.25", rf2("F_python"), 0.2520)
ck_lt("§4 refusal Qwen2 E_insulin ~0", rf2("E_insulin"), 0.15)
# Qwen3
rf3 = lambda k: q3["cells"]["instruct"][f"{k}_a0.6"]["refusal"]["median"]
ck("§4 refusal Qwen3 E_insulin +0.43", rf3("E_insulin"), 0.4298)
ck("§4 refusal Qwen3 F_python -2.44", rf3("F_python"), -2.4426)
ck("§4 refusal Qwen3 A_eiffel -1.34", rf3("A_eiffel"), -1.3429)

# --- §4 P1b Hedge feld-artig ---
hd2 = lambda k: q2["cells"]["instruct"][f"{k}_a0.6"]["hedge"]["median"]
ck("§4 hedge Qwen2 E_insulin -2.35", hd2("E_insulin"), -2.3490)
ck("§4 hedge Qwen2 F_python -1.93", hd2("F_python"), -1.9297)
ck("§4 hedge Qwen2 A_eiffel -1.84", hd2("A_eiffel"), -1.8359)
for k, v in zip(("E_insulin", "F_python", "A_eiffel"), (-0.613, -0.680, -0.775)):
    ck(f"§4 hedge Qwen3 {k}", q3["cells"]["instruct"][f"{k}_a0.6"]["hedge"]["median"], v)

# --- §4 P3 Base hedge (Qwen2 base a0.6) ---
hb2 = lambda k: q2["cells"]["base"][f"{k}_a0.6"]["hedge"]["median"]
ck("§4 hedge BASE E_insulin -2.02", hb2("E_insulin"), -2.0181)
ck("§4 hedge BASE F_python -1.55", hb2("F_python"), -1.5523)
ck("§4 hedge BASE A_eiffel -1.94", hb2("A_eiffel"), -1.9385)

# --- §5 P4a dosis-generisch: hedge + length ratios ---
for r in pl["criteria"]["P4a_hedge"]["cells"]:
    exp = {"E_insulin": 1.116, "F_python": 1.385, "A_eiffel": 0.959}[r["concept"]]
    ck(f"§5 P4a hedge ratio {r['concept']}", r["ratio_p_over_e"], exp)
for r in pl["criteria"]["report_answer_len"]:
    exp = {"E_insulin": 1.157, "F_python": 1.163, "A_eiffel": 1.214}[r["concept"]]
    ck(f"§5 P4a length ratio {r['concept']}", r["ratio"], exp)

# --- §5 P4b Qwen2: E_ai Eiffel engram vs placebo (klarste Zelle) ---
ck("§5 P4b Qwen2 E_ai Eiffel engram +7.78", tp["cells"]["A_eiffel_E_ai_mean"]["engram"]["median"], 7.7836)
ck("§5 P4b Qwen2 E_ai Eiffel placebo -0.05", tp["cells"]["A_eiffel_E_ai_mean"]["placebo"]["median"], -0.0458)

# --- §5 P4b Qwen3: E_ai engram + placebo je Konzept ---
for k, (e, p) in {"E_insulin": (21.13, 1.51), "F_python": (20.93, 9.89), "A_eiffel": (5.33, 4.33)}.items():
    ck(f"§5 P4b Qwen3 E_ai {k} engram", tq["cells"][f"{k}_E_ai_mean"]["engram"]["median"], e, tol=1e-2)
    ck(f"§5 P4b Qwen3 E_ai {k} placebo", tq["cells"][f"{k}_E_ai_mean"]["placebo"]["median"], p, tol=1e-2)
# var_sv Qwen3 dosis-generisch (Placebo staerker): edit_specific_count == 0
ck("§5 P4b Qwen3 var_sv edit_specific=0", tq["criteria"]["var_sv_ai"]["edit_specific_count"], 0)
ck("§5 P4b Qwen3 E_ai edit_specific=2", tq["criteria"]["E_ai_mean"]["edit_specific_count"], 2)

# --- §5 surgical: retain-NLL fairness_ratio 1.8-3.1x (traegt Abstract-Claim) ---
frs=[pl["cells"]["instruct"][f"{k}_a0.6"]["fairness_ratio"] for k in ("E_insulin","F_python","A_eiffel")]
ck("§5 retain-NLL min 1.8x", min(frs), 1.826)
ck("§5 retain-NLL max 3.1x", max(frs), 3.143)
# --- §5 P4b Qwen2 E_ai edit_specific=3 (Gegenstueck zu Qwen3=2) ---
ck("§5 P4b Qwen2 E_ai edit_specific=3", tp["criteria"]["E_ai_mean"]["edit_specific_count"], 3)

# --- §5 P4b Qwen2 symmetrische Zellen-Angaben (Kalibrier-Edit 10.07.) ---
for k, (e, ep, p) in {"E_insulin": (2.30, 0.131, -0.11), "F_python": (3.79, 0.002, 1.60)}.items():
    c = tp["cells"][f"{k}_E_ai_mean"]
    ck(f"§5 P4b Qwen2 E_ai {k} engram", c["engram"]["median"], e, tol=1e-2)
    ck(f"§5 P4b Qwen2 E_ai {k} engram-p", c["engram"]["p"], ep, tol=1e-3)
    ck(f"§5 P4b Qwen2 E_ai {k} placebo", c["placebo"]["median"], p, tol=1e-2)
rat = {r["concept"]: r["ratio_p_over_e"] for r in tp["criteria"]["E_ai_mean"]["cells"]}
for k, v in {"E_insulin": 0.048, "F_python": 0.423, "A_eiffel": 0.006}.items():
    ck(f"§5 P4b Qwen2 ratio {k}", rat[k], v, tol=1e-3)

# --- §5/§7 Robustheits-Zahlen (report-only, results_p2_robustness.json) ---
rb = J("results_p2_robustness.json")
ck("§5 Q3 insulin engram dLen +32.5", rb["qwen3_len"]["E_insulin_engram"]["med_dlen"], 32.5, tol=0.1)
ck("§5 Q3 insulin engram dE_ai +21.13", rb["qwen3_len"]["E_insulin_engram"]["med_dE_ai"], 21.13, tol=1e-2)
ck("§5 Q3 insulin placebo dLen -195", rb["qwen3_len"]["E_insulin_placebo"]["med_dlen"], -194.5, tol=0.6)
ck("§5 Q3 insulin engram 10/10", rb["qwen3_len"]["E_insulin_engram"]["n_pos_dE_ai"], 10)
ck("§5 Q2 eiffel engram 10/10", rb["sign_counts"]["Q2_A_eiffel_engram"]["n_pos"], 10)
ck("§5 Q2 eiffel placebo 4/10", rb["sign_counts"]["Q2_A_eiffel_placebo"]["n_pos"], 4)
ck("§5 sensor Q2 insulin mean_len 97", rb["sensor_capture"]["Q2_E_insulin_engram"]["mean_len"], 97)
ck("§5 sensor Q2 insulin min 4", rb["sensor_capture"]["Q2_E_insulin_engram"]["min_dialog_mean_len"], 4)
ck("§5 sensor Q2 baseline 409", rb["sensor_capture"]["Q2_baseline_mean_len"], 409)
t2e = [rb["sign_counts"][f"Q2_{k}_engram"]["max_abs_dE_ai"] for k in ("E_insulin", "F_python", "A_eiffel")]
t2p = [rb["sign_counts"][f"Q2_{k}_placebo"]["max_abs_dE_ai"] for k in ("E_insulin", "F_python", "A_eiffel")]
t3e = [rb["sign_counts"][f"Q3_{k}_engram"]["max_abs_dE_ai"] for k in ("E_insulin", "F_python", "A_eiffel")]
t3p = [rb["sign_counts"][f"Q3_{k}_placebo"]["max_abs_dE_ai"] for k in ("E_insulin", "F_python", "A_eiffel")]
ck("§7 tails Q2 engram min +18", min(t2e), 18.3, tol=0.1)
ck("§7 tails Q2 engram max +57", max(t2e), 57.2, tol=0.1)
ck("§7 tails Q2 placebo max +8.9", max(t2p, key=abs), 8.9, tol=0.1)
ck("§7 tails Q3 engram max +1546", max(t3e), 1545.7, tol=0.1)
ck("§7 tails Q3 placebo max +217", max(t3p), 217.2, tol=0.1)
checks.append(("§7 tails edit-seitig je Zelle (Q2+Q3)", "per-cell",
               "engram>placebo", all(e > p for e, p in zip(t2e + t3e, t2p + t3p))))
# --- §5 'Not length-mediated': Q2-Eiffel-Laengen (clean cell) ---
q2len_e = rb["sign_counts"]  # Laengen stehen nicht in sign_counts; direkt aus Roh-JSON:
tr_raw = J("results_p2_trajectories.json"); pl_raw = J("results_p2_traj_placebo.json")
mlen = lambda d: sum(len(t) for t in d["ai_texts"]) / len(d["ai_texts"])
med = lambda x: sorted(x)[len(x) // 2] if len(x) % 2 else sum(sorted(x)[len(x) // 2 - 1:len(x) // 2 + 1]) / 2
eng_eif = next(c for c in tr_raw["cells"] if c["alpha"] == 0.6 and c["cut"] == "A_eiffel")
pla_eif = next(c for c in pl_raw["cells"] if c["cut"] == "A_eiffel")
dl_e = med([mlen(eng_eif["dialogs"][i]) - mlen(tr_raw["baseline"][i]) for i in range(10)])
dl_p = med([mlen(pla_eif["dialogs"][i]) - mlen(tr_raw["baseline"][i]) for i in range(10)])
ck("§5 Q2 eiffel engram dLen -180", dl_e, -179.8, tol=0.5)
ck("§5 Q2 eiffel placebo dLen -189", dl_p, -189.4, tol=0.5)

n_ok = sum(c[3] for c in checks)
for name, a, e, ok in checks:
    if not ok:
        print(f"FAIL  {name}: got {a}, expected {e}")
print(f"\n{n_ok}/{len(checks)} PASS")
