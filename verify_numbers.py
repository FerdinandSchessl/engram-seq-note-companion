#!/usr/bin/env python3
"""Checks every number quoted in the note against the shipped raw-result JSONs.

Run: python verify_numbers.py   (stdlib only; expects the JSONs in ./results/)
Expected output: 57/57 PASS.
"""
import json, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
files = {
    "Qwen3-0.6B": "results_seq_c4_Qwen3-0_6B.json",
    "TinyLlama-1.1B": "results_seq_c4_TinyLlama-1_1B-Chat-v1_0.json",
    "Qwen2-0.5B": "results_seq_c4_Qwen2-0_5B-Instruct.json",
}
R = {k: json.load(open(os.path.join(D, f))) for k, f in files.items()}

checks = []
def ck(name, actual, expected, tol=5e-4):
    ok = abs(actual - expected) <= tol
    checks.append((name, actual, expected, ok))

# V1 ratios + edit magnitudes
ck("V1 ratio Qwen3", R["Qwen3-0.6B"]["V1_arm_divergence"]["ratio"], 0.710, 5e-4)
ck("V1 ratio TinyLlama", R["TinyLlama-1.1B"]["V1_arm_divergence"]["ratio"], 0.642, 5e-4)
ck("V1 ratio Qwen2-0.5B", R["Qwen2-0.5B"]["V1_arm_divergence"]["ratio"], 0.614, 5e-4)
ck("dist_m0_m1 Qwen3", R["Qwen3-0.6B"]["V1_arm_divergence"]["dist_m0_m1"], 0.343, 5e-4)
ck("dist_m0_m1 TinyLlama", R["TinyLlama-1.1B"]["V1_arm_divergence"]["dist_m0_m1"], 0.307, 5e-4)
ck("dist_m0_m1 Qwen2-0.5B", R["Qwen2-0.5B"]["V1_arm_divergence"]["dist_m0_m1"], 0.808, 5e-4)
mag_ratio = R["Qwen2-0.5B"]["V1_arm_divergence"]["dist_m0_m1"] / R["Qwen3-0.6B"]["V1_arm_divergence"]["dist_m0_m1"]
ck("magnitude 0.5B/0.6B = 2.4x", mag_ratio, 2.36, 0.05)

# V2
v2 = lambda m, pair, key: R[m]["V2_commutativity"][pair][key]
ck("V2 overlap ratio Qwen3", v2("Qwen3-0.6B", "A_eiffel_B_louvre", "ratio"), 0.997, 5e-4)
ck("V2 overlap maxdNLL Qwen3", v2("Qwen3-0.6B", "A_eiffel_B_louvre", "max_abs_dNLL"), 5.65, 5e-3)
ck("V2 distant ratio Qwen3", v2("Qwen3-0.6B", "A_eiffel_F_python", "ratio"), 0.576, 5e-4)
ck("V2 distant maxdNLL Qwen3", v2("Qwen3-0.6B", "A_eiffel_F_python", "max_abs_dNLL"), 1.63, 5e-3)
ck("V2 overlap ratio TinyLlama", v2("TinyLlama-1.1B", "A_eiffel_B_louvre", "ratio"), 0.912, 5e-4)
ck("V2 overlap maxdNLL TinyLlama", v2("TinyLlama-1.1B", "A_eiffel_B_louvre", "max_abs_dNLL"), 2.48, 5e-3)
ck("V2 distant ratio TinyLlama", v2("TinyLlama-1.1B", "A_eiffel_F_python", "ratio"), 0.466, 5e-4)
ck("V2 distant maxdNLL TinyLlama", v2("TinyLlama-1.1B", "A_eiffel_F_python", "max_abs_dNLL"), 1.15, 5e-3)
ck("V2 overlap ratio Qwen2", v2("Qwen2-0.5B", "A_eiffel_B_louvre", "ratio"), 0.809, 5e-4)
ck("V2 overlap maxdNLL Qwen2", v2("Qwen2-0.5B", "A_eiffel_B_louvre", "max_abs_dNLL"), 9.57, 5e-3)
ck("V2 distant ratio Qwen2", v2("Qwen2-0.5B", "A_eiffel_F_python", "ratio"), 0.677, 5e-4)
ck("V2 distant maxdNLL Qwen2", v2("Qwen2-0.5B", "A_eiffel_F_python", "max_abs_dNLL"), 5.99, 5e-3)
# V2 third-concept detail (Qwen3): order of the Paris pair decides the Colosseum's fate
ck("V2 C-NLL xy Qwen3", v2("Qwen3-0.6B", "A_eiffel_B_louvre", "nll_xy")["C_colosseum"], 0.85, 5e-3)
ck("V2 C-NLL yx Qwen3", v2("Qwen3-0.6B", "A_eiffel_B_louvre", "nll_yx")["C_colosseum"], 6.50, 5e-3)

# V3: dNLL after the A cut (Arm 1 step 1 vs nll_m0)
def v3(m):
    base, s1 = R[m]["nll_m0"], R[m]["arm1_nll_steps"][0]["nll"]
    return {k: s1[k] - base[k] for k in base}
d1, d2, d3 = v3("Qwen3-0.6B"), v3("TinyLlama-1.1B"), v3("Qwen2-0.5B")
ck("V3 Qwen3 B", d1["B_louvre"], 1.57, 5e-3)
ck("V3 Qwen3 D", d1["D_brandenburg"], 0.76, 5e-3)
ck("V3 Qwen3 C", d1["C_colosseum"], 0.08, 5e-3)
ck("V3 Qwen3 F", d1["F_python"], 0.08, 5e-2)
ck("V3 TinyLlama B", d2["B_louvre"], 0.96, 5e-3)
ck("V3 Qwen2 C", d3["C_colosseum"], 8.47, 5e-2)
ck("V3 Qwen2 D", d3["D_brandenburg"], 5.58, 5e-2)
ck("V3 Qwen2 F", d3["F_python"], 5.44, 5e-2)
ck("V3 Qwen2 B", d3["B_louvre"], 3.77, 5e-2)
ordering_fail = not (d3["B_louvre"] == max(d3[k] for k in d3 if k != "A_eiffel"))
checks.append(("V3 Qwen2 broken (B not top)", ordering_fail, True, ordering_fail is True))

# V4: strict monotonicity of ALL survivor-drift series (>=2 points)
n_series, n_mono, three_pt = 0, 0, 0
for m in R:
    steps = R[m]["arm2_steps"]
    concepts = set().union(*[s["cov_drift_remaining"].keys() for s in steps])
    for c in sorted(concepts):
        series = [s["cov_drift_remaining"][c] for s in steps if c in s["cov_drift_remaining"]]
        if len(series) >= 2:
            n_series += 1
            if len(series) == 3: three_pt += 1
            if all(b > a for a, b in zip(series, series[1:])):
                n_mono += 1
print(f"V4 monotonicity: {n_mono}/{n_series} series strictly increasing ({three_pt} with 3 points)")
checks.append(("V4 all series monotone", n_mono, n_series, n_mono == n_series))

# V4 endpoint values (note table)
for m, exp in [("Qwen3-0.6B", {"B_louvre": (1.64, 1.80), "D_brandenburg": (None, 0.49), "F_python": (None, 0.23)}),
               ("TinyLlama-1.1B", {"B_louvre": (0.41, 0.79), "D_brandenburg": (None, 0.64), "F_python": (None, 0.45)}),
               ("Qwen2-0.5B", {"B_louvre": (1.24, 2.45), "D_brandenburg": (None, 2.04), "F_python": (None, 1.95)})]:
    steps = R[m]["arm2_steps"]
    for c, (first, last) in exp.items():
        if first is not None:
            ck(f"V4 {m} {c} start", steps[0]["cov_drift_remaining"][c], first, 5e-3)
        ck(f"V4 {m} {c} end", steps[-1]["cov_drift_remaining"][c], last, 5e-3)

# Bonus: partial return of the erased concept A under subsequent cuts
a1 = [s["nll"]["A_eiffel"] for s in R["Qwen3-0.6B"]["arm2_steps"]]
ck("bonus Qwen3 A step2", a1[1], 13.20, 5e-2)
ck("bonus Qwen3 A step3", a1[2], 9.88, 5e-2)
a3 = [s["nll"]["A_eiffel"] for s in R["Qwen2-0.5B"]["arm2_steps"]]
ck("bonus Qwen2 A step1", a3[0], 8.91, 5e-2)
ck("bonus Qwen2 A step2", a3[1], 5.21, 5e-2)
ck("bonus Qwen2 A step3", a3[2], 4.85, 5e-2)
ck("bonus Qwen2 baseline A", R["Qwen2-0.5B"]["nll_m0"]["A_eiffel"], 0.05, 5e-3)
a2 = [s["nll"]["A_eiffel"] for s in R["TinyLlama-1.1B"]["arm2_steps"]]
ck("bonus TinyLlama weak |step3-step2|", abs(a2[2] - a2[1]), 0.25, 5e-2)

# Louvre collateral, Arm 1 vs Arm 2 final state (Qwen3)
ck("Louvre Arm1 final Qwen3", R["Qwen3-0.6B"]["arm1_nll_steps"][-1]["nll"]["B_louvre"], 2.82, 5e-3)
ck("Louvre Arm2 final Qwen3", R["Qwen3-0.6B"]["arm2_steps"][-1]["nll"]["B_louvre"], 6.93, 5e-3)

# Insulin rise after the Colosseum cut, Arm 1 (Qwen2-0.5B): 2.2 -> 15.2 (+13)
e_step1 = R["Qwen2-0.5B"]["arm1_nll_steps"][0]["nll"]["E_insulin"]
e_step2 = R["Qwen2-0.5B"]["arm1_nll_steps"][1]["nll"]["E_insulin"]
print(f"insulin Arm1 Qwen2-0.5B: step1={e_step1} -> step2={e_step2}, delta={e_step2-e_step1:.2f} (baseline {R['Qwen2-0.5B']['nll_m0']['E_insulin']})")

# TinyLlama landmark baselines 5.7-6.3 nats (weak anchoring)
tl_base = R["TinyLlama-1.1B"]["nll_m0"]
landmarks = [tl_base[k] for k in ["A_eiffel", "B_louvre", "C_colosseum", "D_brandenburg"]]
checks.append(("TinyLlama landmark baselines in [5.7,6.3]", round(min(landmarks),2), round(max(landmarks),2), 5.7 <= min(landmarks) and max(landmarks) <= 6.3))

# Runtimes
ck("runtime Qwen3", R["Qwen3-0.6B"]["runtime_sec"], 2221.6, 1)
ck("runtime TinyLlama", R["TinyLlama-1.1B"]["runtime_sec"], 15617.5, 1)
ck("runtime Qwen2", R["Qwen2-0.5B"]["runtime_sec"], 5011.4, 1)

fails = [c for c in checks if not c[3]]
for name, a, e, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: actual={a} expected={e}")
print(f"\n=== {len(checks)-len(fails)}/{len(checks)} PASS, {len(fails)} FAIL ===")
raise SystemExit(1 if fails else 0)
