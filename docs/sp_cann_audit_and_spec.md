# SP-CANN: Codebase Audit and Sprint Spec

**Date: April 2026**

This document records the results of a full audit of the existing codebase against
the SP-CANN architecture proposed in `sp_cann_discussion.md`, and translates the
gap analysis into a prioritised, sprint-ready todo-list specification.

---

## Part 1 — Audit: What We Have vs. What We Need

### ✅ Confirmed Working (no changes needed)

| Component | Location | Status |
|-----------|----------|--------|
| FFNN baseline | `models/ffnn.py` | 98.9%+ |
| CNN baseline | `models/cnn.py` | 99.17% ± 0.10% |
| SNN-FFNN (SNNBaseline) | `models/snn_baseline.py` | 97.62% ± 0.12% |
| SNN-CNN | `models/snn_cnn.py` | 98.87% ± 0.13% |
| PC-FFNN v3 | `models/pc_ffnn.py` | 97.30% ± 0.22% |
| PC-EncDec v2 (step 6a) | `models/pc_enc_dec.py` | 97.14% ± 0.21% |
| PC two-phase training infrastructure | `train_pcnn.py` | Working |
| PC free energy F (encoder + decoder) | all `pc_*.py` | Working |
| PC inference loop (`_run_inference`) | all `pc_*.py` | Working, T_pc=20 |
| Rate & TTFS spike encoding | `utils/spike_encoding.py` | Working |
| Layer activation recorder | `record.py` | Partial — see below |
| MNIST-Video dataset | `datasets/mnist_video.py` | Working |
| Hydra config + multi-seed CHPC pipeline | `configs/`, `scripts/` | Working |

### ⚠️ Exists but Broken / Limited / Needs Rework

| Component | Issue |
|-----------|-------|
| `spc_ffnn.py` (SPC-FFNN v1, iters 2–5) | All 4 variants at chance (~11%). Root cause confirmed: shared SNN/PC weight matrices. Architecture is a dead end. Code stays as documented failure; do not reuse. |
| `record.py` | Hardcoded to `SNNBaseline.forward_with_recordings()` with fixed layer keys (`spk1/mem1/spk2/mem2/spk3/mem3`). Cannot record from CNN, PC-CNN, or arbitrary source models. Needs generalisation. |
| PC-CNN (iter 11) | 93.12% ± 0.56% — underperforms all baselines. Stabilised variant (iter 12, PC-CNN v2) in progress. Await results before using as source network. |
| SNN recurrent models (`anp_snn`) | SRNN and SLSTM not yet implemented. RNN baselines (iters 3–10) are hybrid, not fully spiking. |

### ❌ Does Not Exist — Needs to Be Built

| Component | SP-CANN Step | Notes |
|-----------|-------------|-------|
| Generalised activation extractor | Step 4 | Generic `forward_with_recordings` protocol or hook-based extractor for any source model |
| Source network freezer + all-layer concat | Step 4 | Freeze source, collect `a = concat(all layer activations)`, shape `(N,)` |
| Population interface / artificial EEG | Step 5 | `e = W_pop @ a + ε`, fixed random `W_pop ∈ ℝ^{C×N}`, Gaussian noise, sweep C ∈ {16, 64, 128} |
| Artificial EEG dataset builder | Steps 3/5 | Loop 2 data: run frozen source on MNIST, store `{e_t}` tensors |
| **Separate-weight SPC architecture** (iter 6) | Step 7 | SNN encoder with its own weights → spike_accs → cls_head (CE+surrogate). PC generative model with **separate** weights. Lesson from iters 2–5. |
| Error column encoder (ε_enc) | Step 6b | Encodes `ε^l = r^{l-1} − r̂^{l-1}` signals bottom-up into latent space |
| Shared latent space z | Step 6b | Bottleneck merging `r_enc` and `ε_enc` from both columns |
| Stimulation decoder (ε_dec → s_out) | Step 6b/9 | Decodes z back through error column: `s_out = −α·ε_dec` |
| Skip connections (U-Net style) | Step 6b | `r_in → r_dec` and `ε_in → ε_dec` preserving fine-grained interface signal |
| Reciprocal connections | Step 6b | Representation ↔ error column bidirectional exchange at each level |
| Loop 2 training harness | Step 6 | Self-supervised training of auxiliary on `{e_t}` to minimise F, no labels |
| Decay simulation module | Step 8 | Hard ablation: `a_decayed = m ⊙ a`, `m ~ Bernoulli(1−d)`, fixed per seed |
| ΔF anomaly detection metric | Step 8 | `ΔF_t = F(e_t) − F̄_normal > θ → decay detected` |
| Layer-wise `\|ε^l\|` localisation | Step 8 | Per-layer error magnitudes to localise where in source drift occurred |
| Three-condition evaluation loop | Step 9 | For each d: run source alone (acc_decay), source+AUX detection (ΔF), source+AUX stimulation on (acc_restored) |
| Stimulation injection into source | Step 9 | Add `s_out` additively at the population interface. Does **not** modify source weights. |
| Recovery ratio ρ metric | Step 9 | `ρ = (acc_restored − acc_decay) / (acc_intact − acc_decay)` per (seed, d) |
| SNN auxiliary (step 7) | Step 7 | Replace dense AUX neurons with LIF. SNN encoder and PC generative weights must be **separate**. |

---

## Part 2 — Sprint Spec (Ordered by Critical Path)

### S4 — Source + AUX Split

**Goal:** establish the infrastructure to freeze a source network and extract a
flat activation vector from all its layers.

- [ ] **Generalise `record.py`** to support arbitrary source models via `nn.Module`
  hooks rather than hardcoded `forward_with_recordings`. All layer activations
  captured and concatenated to vector `a`.
- [ ] **Implement source network freezer** — utility `freeze_source(model)` that
  sets all params `requires_grad=False` and returns the frozen model.
- [ ] **Implement all-layer activation extractor** — `extract_activations(model, x) → Tensor[N_neurons]`
  using forward hooks, handles CNN (flatten spatial), FFNN, and SNN (spike-count)
  sources uniformly.
- [ ] **Validate on CNN source** — load best CNN checkpoint, run extractor on MNIST
  test set, verify concatenated `a` shape is correct.

---

### S5 — Population Interface (Bandwidth Constraint)

**Goal:** impose the biological bandwidth constraint by compressing the full
activation vector through a fixed random projection.

- [ ] **Implement `PopulationInterface` module** — fixed random `W_pop ∈ ℝ^{C×N}`
  (not trained), additive Gaussian noise `ε ~ N(0, σ²I)`, outputs `e = W_pop @ a + ε`.
  `C` and `σ` are config params.
- [ ] **Build artificial EEG dataset** — script to run frozen source on full MNIST
  training set, apply `PopulationInterface`, save `{e_t, label_t}` pairs as a
  `torch.Dataset`. Output to `outputs/aeeg/{experiment_group}/{C_channels}/{seed}/`.
- [ ] **Sweep C ∈ {16, 64, 128}** — generate three dataset variants, validate mutual
  information `I(e; y)` by training a linear probe on `e` and comparing to the
  full-activation baseline.
- [ ] **Config support** — add `configs/dataset/aeeg.yaml` and
  `configs/model/population_interface.yaml`.

---

### S6 — PC-AUX Representation Column (Step 6a Validation)

**Goal:** confirm that `PCEncDec` can serve as an auxiliary operating on `e_t`
(compressed channel vector, not raw images), and establish the Loop 2 training
harness.

- [ ] **Audit `PCEncDec` as auxiliary** — verify it can be used with `input_size=C`
  (not 784). Confirm it can train self-supervised on `{e_t}` without labels.
- [ ] **Implement Loop 2 training script** (`train_auxiliary.py` or mode flag in
  `train_pcnn.py`) — self-supervised, no labels, trains `PCEncDec` on `{e_t}` to
  minimise F. Logs F on held-out normal `e` for convergence criterion. No CE head.
- [ ] **Validate decoding readout** — add `r_out → task_label` linear classifier
  head on top of `r_out` from the frozen auxiliary, train with CE. Report
  `acc_decode` as RQ1 baseline.
- [ ] **Verify loop** — smoke test: AUX trained on `C=128` artificial EEG from CNN
  source, linear probe decoding accuracy should be > chance.

---

### S7 — Error Column + Shared Latent (Step 6b — Core Novelty)

**Goal:** implement the SP-CANN four-quadrant architecture with the error column,
shared latent space, and stimulation output.

- [ ] **Design `SPCANN` class** (`models/sp_cann.py`) with four-quadrant structure:
  - Representation column: `r_in → r_enc` (encoder) + `z → r_dec → r_out` (decoder)
  - Error column: `ε_in → ε_enc` (encoder) + `z → ε_dec → s_out` (decoder)
  - Shared latent: `z = f(r_enc, ε_enc)` — bottleneck
  - Skip connections: `r_in` to `r_dec`, `ε_in` to `ε_dec`
- [ ] **Implement ε computation** — at each level `l`:
  `ε^l = r^{l-1} − r̂^{l-1}` (prediction error from representation column decoder)
- [ ] **Implement stimulation output** — `s_out = −α · ε_dec`, `α` fixed scalar
  (first experiment), `s_out` has same shape as `e`
- [ ] **Enforce separate weight matrices** — SNN encoder weights (`snn_enc_*`) must
  be entirely separate from PC generative weights (`pc_gen_*`). No sharing.
  This is the invariant lesson from anp_spcnn iters 2–5.
- [ ] **Unit test** — confirm forward pass produces `r_out` (task readout) and
  `s_out` (stimulation) with correct shapes. Confirm F decreases over inference steps.
- [ ] **Config** — `configs/model/sp_cann.yaml`,
  `configs/experiment/mnist_sp_cann_loop2.yaml`.

---

### S8 — Decay Simulation (Step 8)

**Goal:** introduce controlled neuron ablation into the source network and establish
the anomaly detection baseline.

- [ ] **Implement `NeuronAblation` wrapper** — `AblationWrapper(source_model, d, seed)`
  that registers forward hooks zeroing `d%` of neurons. Mask `m ~ Bernoulli(1−d)`
  fixed per seed. Supports `d ∈ {0, 10, 20, 30, 40, 50}`.
- [ ] **Verify measurable performance drop** — run ablated CNN source (no AUX) on
  MNIST test set across d levels and 5 seeds. Record `acc_decay(d)` baseline.
  Confirm ≥5 pp drop at d=30%.
- [ ] **Implement F̄_normal baseline** — run auxiliary (trained in Loop 2) on normal
  `{e_t}` (d=0), compute mean `F̄_normal` and std.
- [ ] **Implement ΔF anomaly detector** — `detect_decay(aux, e_t, F_normal_baseline) → bool`.
  Compute `ΔF_t = F(e_t) − F̄_normal`. Threshold `θ` as a config param (default = 2σ).
- [ ] **Layer-wise ε^l logging** — extend auxiliary `forward()` to return per-layer
  `|ε^l|` alongside overall F. Log to Wandb.
- [ ] **Experiment: detection accuracy** — report AUROC of `ΔF_t` as anomaly
  detector across d levels.

---

### S9 — Stimulation Loop (Step 9 — RQ2)

**Goal:** close the loop: inject the auxiliary's stimulation signal back into the
source and measure functional restoration.

- [ ] **Implement three-condition evaluation harness** —
  `eval_three_conditions(source, aux, dataloader, d, seed)` returns
  `{acc_intact, acc_decay, acc_restored, delta_F, rho}`. This is the primary
  empirical protocol.
- [ ] **Implement stimulation injection** — `e_stimulated = e + s_out` where
  `s_out = −α·ε_dec`. Source receives stimulated `e` instead of raw `e`.
  Source weights are unchanged.
- [ ] **Implement recovery ratio ρ** —
  `ρ = (acc_restored − acc_decay) / (acc_intact − acc_decay)`.
- [ ] **Run ablation grid** — jobs for every `(C, d)` cell in the ablation table.
  5 seeds each. Embarrassingly parallel on CHPC.
- [ ] **Statistical test** — Wilcoxon signed-rank on `acc_restored > acc_decay`
  across seeds per d. Report p-values.

---

### S10 — SNN Auxiliary (Step 7, Post-S9 Baseline)

**Goal:** replace dense auxiliary neurons with LIF, making the auxiliary itself
biologically plausible.

- [ ] **Replace dense AUX layers with `snn.Leaky`** — representation and error
  column neurons become LIF. PC generative weights remain separate from SNN
  encoder weights (invariant from S7).
- [ ] **Rate-coded representation** — `r̄^l = (1/T) Σ_t s^l_t` for PC weight
  update rule (as specified in §11.2 of the discussion document).
- [ ] **Validate parity** — SNN-AUX should not significantly underperform dense AUX
  on decoding accuracy (RQ1). If > 2 pp drop, investigate T and β.

---

## Part 3 — Infrastructure Debt (Needed Across All Sprints)

- [ ] **Generalise `record.py`** — hook-based, model-agnostic (blocks S4)
- [ ] **PC-CNN v2 results** — analyse iter 12 CHPC results once complete; update
  leaderboard; decide if PC-CNN v2 becomes the source network or CNN stays
- [ ] **Pending anp_snn iters 11–14** — SRNN and SLSTM needed before S10 but not
  blocking S4–S9
- [ ] **Artificial EEG dataset validation** — `anp_datasets` iter 3 is the same
  work as S5 above; the two tracks should be merged to avoid duplication
- [ ] **`train_auxiliary.py`** — a third training script is needed (Loop 2, no CE,
  self-supervised F minimisation). Can be a mode flag in `train_pcnn.py` instead.

---

## Part 4 — What Is Notably Absent from the Codebase

A quick summary of the six most critical missing pieces, for sprint planning
prioritisation:

1. **No population interface code** — `W_pop`, artificial EEG generation, channel
   sweep. Nothing exists.
2. **No source-freezing + activation extraction** — `record.py` is too narrow
   (SNNBaseline-only, fixed layer keys).
3. **No error column** — `PCEncDec` has a reconstruction decoder but not an
   error-signal encoder → latent → stimulation pathway.
4. **No stimulation injection** — the additive `s_out` path into the source
   interface does not exist anywhere.
5. **No decay/ablation simulation** — no `AblationWrapper`, no ΔF computation,
   no `acc_decay` baseline protocol.
6. **No three-condition evaluation harness** — the intact/decay/restored comparison
   protocol does not exist.
