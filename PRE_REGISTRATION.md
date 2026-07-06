# Pre-Registration — Sequential C.4 (Path Dependence in Sequential Engram Editing)

**Provenance.** The test design and predictions below were fixed in a private research
log on **2026-07-05, prior to any measurement**, as part of a reading analysis of
Kwon et al., "AI Engram: In Search of Memory Traces in Artificial Intelligence"
(arXiv:2606.14997, ICML 2026 oral). They are reproduced here verbatim (translated
from German). Independent timestamping (e.g. OSF) was not used; this is an internal
pre-registration, disclosed for transparency. The operationalization (two-arm design,
V1–V4) was fixed in the header of the measurement script on the same day, before the
first full run; it is quoted in §2 below and unchanged in `seq_c4_test.py`.

## 1. Design (research log, 2026-07-05, verbatim)

Run their C.4 protocol sequentially (their code is public):

1. k = 1…N engram cuts **on the same model** (not each on a fresh copy), target
   concepts with graded semantic overlap.
2. After each cut, measure: (i) same-neighborhood collateral as in their C.4;
   (ii) covariance drift of the survivors ‖Σ_i^(k) − Σ_i^(0)‖ (their own sufficient
   statistics used as strain gauges); (iii) behavioral metrics.
3. Commutativity test: Engram(A)∘Engram(B) vs. Engram(B)∘Engram(A) — weight AND
   behavioral distance.
4. Optional recovery cycle (brief retain fine-tuning) after each cut: does Σ return
   (elastic) or drift further (plastic)?

**Predictions:** drift > 0, structured (proximity-monotone as in their C.4),
**accumulating in k** (super-linear if superposition is real); A∘B ≠ B∘A growing
with concept overlap; everything stronger in the more redundant substrate.
A null result would be equally informative: then their linearity is elastic under
sequential load as well — strong support for their Appendix F, to be reported honestly.

## 2. Operationalization (script header, 2026-07-05, verbatim)

Two arms per model, package `ai-engram` (their reference implementation):

- Arm 1 (their assumption): extract all engrams on M0, apply in sequence (linear).
- Arm 2 (re-calibrated): after each cut, re-extract on the CURRENT state.

- **V1:** Arm1 ≠ Arm2 (weight/NLL difference ≫ 0), i.e. superposition/redistribution is real.
- **V2:** Non-commutativity (Arm-2 style): dist(M_AB, M_BA) > dist(M_AF, M_FA) [overlap > distant].
- **V3:** Collateral after the A cut is proximity-structured: dNLL(B) > dNLL(E), dNLL(F).
- **V4:** Covariance drift of uncut concepts > 0, proximity-structured, accumulating.

A null result is informative (supports their App. F) and is reported just the same.

## 3. Deviations of the executed experiment from the log design

Reported for honesty; none was motivated by interim results:

- **Substrate.** The log names their CIFAR-100/ResNet-18 track as the CPU-cheap
  option and the more redundant LLM substrate as the stronger test. The experiment
  was run on the LLM track (their quickstart model Qwen3-0.6B, then two replication
  charges), not on CIFAR/ResNet. The "stronger in the more redundant substrate"
  prediction is therefore untested (no cross-substrate comparison was run).
- **Behavioral metrics (design point 2.iii)** were operationalized as teacher-forced
  NLL on held-out cloze probes per concept — the narrowest behavioral readout; no
  free-generation metrics were collected.
- **Recovery cycles (design point 4)** were not run (marked optional in the design;
  designated follow-up).
- **Replication charges** (TinyLlama-1.1B-Chat-v1.0, Qwen2-0.5B-Instruct) were added
  after the first charge confirmed V1–V4, with the identical protocol and unchanged
  predictions. A fourth charge (Qwen2-1.5B-Instruct) failed twice with kernel OOM
  kills during Arm 1 on a 128-GB host — an infrastructure failure, not a finding;
  the smallest-capacity substitute (Qwen2-0.5B-Instruct) was used instead. Their
  TOFU substrate (Llama-3.2-1B) is gated on Hugging Face and was not accessible.
- **α = 0.6** is the value the authors selected for their fixed-α Engram variant on
  the TOFU LLM benchmark (Table 3; grid search {0.05..1.0}), and was fixed before the
  first full run. It is their own value, adopted rather than tuned to amplify the
  effect (a smoke test at α = 1.0 saturated collateral across the board); α was not
  swept toward the small-α limit where first-order composition is most accurate.
