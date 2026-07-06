# Companion — "Forgetting Is Not a Fix: Path Dependence in Sequential Engram Editing"

Code and raw data for the note *"Forgetting Is Not a Fix: Path Dependence in
Sequential Engram Editing"* (Ferdinand M. Schessl, 2026; arXiv ID to be added
after announcement).

The note tests the Compositional Memory States Hypothesis (Appendix F) of

> Jea Kwon, Dong-Kyum Kim, Jiwon Kim, Yonghyun Kim, Woong Kook, Meeyoung Cha:
> "AI Engram: In Search of Memory Traces in Artificial Intelligence."
> arXiv:2606.14997, ICML 2026 (oral). Code: <https://github.com/jeakwon/ai-engram>

under *sequential* load: engram cuts applied one after another to the same model,
with pre-registered predictions (V1–V4), across three model charges. Everything
runs on the authors' own reference implementation (`ai-engram`, PyPI) at their
reported TOFU-best edit strength (α = 0.6) — a choice deliberately in favor of
the linearity hypothesis being tested.

## Contents

| File | What it is | Note section |
|---|---|---|
| `seq_c4_test.py` | The complete experiment (one script, `--model` switch, two arms, V1–V4) | §2–§4 |
| `PRE_REGISTRATION.md` | Design + predictions fixed 2026-07-05 before measurement, verbatim, with a deviations section | §2, §6 |
| `results/results_seq_c4_Qwen3-0_6B.json` | Raw results, charge 1 (their quickstart model) | §4 |
| `results/results_seq_c4_TinyLlama-1_1B-Chat-v1_0.json` | Raw results, charge 2 (Llama architecture, different vendor) | §4 |
| `results/results_seq_c4_Qwen2-0_5B-Instruct.json` | Raw results, charge 3 (smallest capacity) | §4 |
| `verify_numbers.py` | Re-derives every number quoted in the note from the raw JSONs (57 checks) | all |

## Reproduce

```bash
pip install ai-engram==0.8.0        # pulls torch + transformers; CPU is sufficient
python seq_c4_test.py --model Qwen/Qwen3-0.6B
python seq_c4_test.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
python seq_c4_test.py --model Qwen/Qwen2-0.5B-Instruct
```

Deterministic (teacher-forced NLL, closed-form edits, no sampling): re-runs
reproduce the shipped JSONs. Observed CPU runtimes (16 threads): ≈ 2 200 s /
15 600 s / 5 000 s per charge. RAM note: models ≥ 1.5 B parameters need
substantial memory for the MLP input covariances during statistics collection —
Qwen2-1.5B-Instruct was OOM-killed twice on a 128-GB host during Arm 1 (documented
infrastructure failure, not a finding).

To check the note's numbers against the shipped raw data (stdlib only):

```bash
python verify_numbers.py    # expected: 57/57 PASS
```

## What the two arms mean

- **Arm 1 — zero-shot composition (their Appendix-F reading):** all engrams are
  extracted once on the virgin model M0 and applied in sequence as linear
  arithmetic. Extraction ignores load history by construction.
- **Arm 2 — re-calibrated sequence:** after each cut, the next engram is extracted
  on the *current* model state. This is what sequential deployment actually does.

If Appendix F held, the arms would coincide, cut order would be interchangeable,
and the survivors' statistics would stay put. The measured outcome (all three
charges): the arms diverge by 61–71 % of the total edit magnitude, cut order
matters in proportion to concept overlap, and the survivors' covariances drift
monotonically with every further cut. See the note for the full result table and
the two boundary findings (V3 capacity break; partial return of erased knowledge
under subsequent cuts).

## Deviations of this companion from the as-run script

The measurement logic is unchanged. Differences, for transparency:

1. As-run, the package was imported from a local checkout via `sys.path`; here it
   is the pinned PyPI package (`ai-engram==0.8.0`, same version).
2. As-run, results were written next to the script, and charge 1 wrote to a
   default filename; here all charges write uniformly to `results/results_seq_c4_<slug>.json`
   (the shipped charge-1 JSON was renamed accordingly, contents untouched).
3. Comments and log strings were translated from German to English.

## License

MIT (this repository). The `ai-engram` package is MIT-licensed by its authors;
this repository contains none of its code, only imports.
