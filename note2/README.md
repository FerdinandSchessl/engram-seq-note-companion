# Companion — Note 2: "Surgical in Knowledge, Generic in Behavior"

Reproducibility companion for Note 2 of the engram series (*Post-Edit Behavioral Qualification of Engram Edits*). Note 1 companion is in the parent repository; this note lives under `note2/`.

## What this is

Note 2 ablates one distant concept at a time from a chat model (the authors' `ai-engram`, fixed α=0.6) and asks which behavioral side-effects are **edit-specific** vs. **dose-generic**, using a dose-matched random-perturbation placebo as the control. Two charges: Qwen2-0.5B (±Instruct) and Qwen3-0.6B.

**Headline:** coarse markers (refusal, hedge, length) are dose-generic (the placebo reproduces them); the activation-energy stiffness E_ai of the model's own dialogue is edit-specific and replicates on both charges; a second trajectory metric (velocity variance) does not replicate.

## Verify every number (stdlib only)

```
python verify_numbers.py      # expects the JSONs in ./results/
```
Expected output: `60/60 PASS`. Re-derives every number quoted in the note from the raw-result JSONs.

## Script ↔ note mapping

| Script (`scripts/`) | Produces | Note section |
|---|---|---|
| `p2_markers.py` + `p2_prompts.json` | refusal/hedge/sycophancy markers, both charges | §4 P1a/P1b/P3 |
| `p2_analyze.py` | marker medians/CIs/criteria | §4 |
| `p2_trajectories.py` + `p2_dialogs.json` | E_ai / Var(v_sem) on retain dialogues (Qwen2) | §5 P2 |
| `p2_placebo.py` | dose-matched placebo, marker battery (P4a) | §5 P4a |
| `p2_analyze_placebo.py` | engram-vs-placebo, fairness gate | §5 P4a |
| `p2_traj_placebo.py` + `p2_analyze_traj_placebo.py` | trajectory placebo, edit-specificity (Qwen2) | §5 P4b |
| `p2_traj_qwen3.py` + `p2_analyze_traj_qwen3.py` | P4b replication (Qwen3) | §5 P4b |
| `p2_len_control.py` | length-sensitivity control | §7 |
| `entlastung_test.py` + `entlastung_analyze.py` | neighborhood revival pre-test (C'→C'' turn) | §7 / §8 outlook |
| `seq_c4_test.py` | shared engram-cut machinery (from Note 1) | — |

## Regime

`ai-engram` unmodified public API, α=0.6 (the authors' TOFU fixed-α value), FP32, CPU, deterministic (teacher-forced logprob markers; greedy trajectory generation — no sampling). Placebo dose-matching is tied-embedding-safe (matches the readout layer). Frozen pre-registration with a deviations section: `PRE_REGISTRATION.md`.

## Install

```
pip install ai-engram==0.8.0 transformers torch sentence-transformers scipy numpy
```
Scripts expect `seq_c4_test.py` (shipped here) importable; the per-script `sys.path` inserts assume this directory layout.

## Disclosure

Phenomena, procedures, effect directions, and everything running on the authors' tools are public. Calibration thresholds for operational use are maintained separately. This companion contains no operational thresholds.
