# SP-CANN: Supervisor Discussion Document

**Arné Schreuder — PhD Discussion Notes**
**Date: April 2026**

---

## 1. The Core Premise

The human brain is simultaneously the model *and* the learning mechanism. The same
neural substrate that executes cognition is the one that learns and adapts. This
creates a catastrophic vulnerability: when neurons die, not only does functional
capacity degrade, but the system's ability to compensate for that loss also degrades.
This is the dual failure mode of neurodegeneration.

This research proposes a computational approach to counteract that first failure —
functional degradation — using a small, learned auxiliary network that interfaces with
a larger source network through a bandwidth-limited connection, detects drift from
normal activity, and delivers corrective stimulation signals to restore or compensate
for lost function.

---

## 2. The Proposed Model: SP-CANN

**Spiking Predictive Coding Auxiliary Neural Network**

The SP-CANN is the final target architecture. It is defined by five simultaneous
constraints:

1. **Auxiliary** — a small network external to and separate from the source network
2. **Bandwidth-limited** — interfaces only through a compressed population signal (artificial EEG)
3. **Spiking** — uses LIF neurons for biological plausibility and energy efficiency
4. **Predictive coding** — learns via local error minimisation, no backpropagation
5. **Bidirectional** — both records (decodes source activity) and stimulates (injects corrective signals)

### 2.1 Architecture: Four-Quadrant Structure

The SP-CANN has a four-quadrant internal structure arising from two columns and two rows:

```
                  Representation column (r)    Error column (ε)
                  ─────────────────────────    ────────────────
Encoder row  │    r_in → r_enc                 ε_in → ε_enc
             │         ↘                            ↙
Latent       │              z = f(r_enc, ε_enc)
             │         ↙                            ↘
Decoder row  │    z → r_dec → r_out            z → ε_dec → s_out
```

**Representation column:** encodes the population signal into a compressed representation,
then decodes back to a task readout. This is the recording pathway (RQ1).

**Error column:** at each level, computes `ε^l = r^{l-1} − r̂^{l-1}`. These errors are
themselves encoded, compressed into the latent space alongside representations, then
decoded back into a stimulation signal. This is the stimulation pathway (RQ2).

**Shared latent space z:** the bottleneck where both columns meet. Jointly encodes
*what the source network is doing* and *how far it has drifted from normal*. Inspired
by VAEs and autoencoders.

**Skip connections:** `r_in` skips to inform `r_dec`, and `ε_in` skips to inform
`ε_dec`. Preserves fine-grained interface information across the bottleneck. Inspired
by U-Net.

**Reciprocal connections:** at each level, the representation and error columns exchange
signals bidirectionally — representations generate top-down predictions, errors
propagate bottom-up mismatches. Standard PC dynamics.

**Recurrent connections:** within-level lateral connections allow representations to
stabilise over time. Inspired by GLOM's column consensus mechanism.

### 2.2 Key Mathematical Definitions

**Population interface (artificial EEG):**

```
e = W_pop · a + ε,   W_pop ∈ ℝ^{C×N},   ε ~ N(0, σ²I)
```

`a ∈ ℝ^N` is the concatenated activation vector across all source network layers.
`W_pop` is a fixed random projection (not learned). `C ≪ N` is the channel count.

**PC energy (free energy proxy):**

```
F = Σ_l  (1/2σ_l²) |ε^l|²,   ε^l = r^{l-1} − r̂^{l-1}
```

The auxiliary network minimises `F` on normal source activity via local weight updates only.

**Anomaly detection:**

```
ΔF_t = F(e_t) − F̄_normal > θ  ⟹  decay detected
```

Layer-wise errors `ε^l` also localise *where* in the source network drift occurred.

**Stimulation signal:**

```
s_out = −α · ε_dec
```

The decoded, expanded error signal is injected additively at the interface — not
modifying source network weights, only its operating point. Analogous to deep brain
stimulation.

**Decoding objective (RQ1):**

```
ŷ = g(r_out),   L_decode = CE(ŷ, y)
```

**Three-condition hypothesis (RQ2):**

For degradation levels `d ∈ {0, 10, 20, 30, 40, 50}%` neurons ablated:

```
acc_intact(d=0) ≥ acc_restored(d) > acc_decay(d)   ∀ d ≤ d*
```

Claim: there exists a `d* > 0` such that the above holds with statistical significance
(`p < 0.05` across seeds). Partial restoration is the publishable bar; full restoration
is the aspiration.

---

## 3. Novelty Argument

### 3.1 What SP-CANN Is Not

| Technique | Why it is different from SP-CANN |
|-----------|----------------------------------|
| Adapter networks / LoRA | White-box, full parameter access, high bandwidth, offline, modifies source weights |
| Knowledge distillation | Unidirectional, offline, requires teacher outputs directly |
| Clinical neural prosthetics | Signal processing only (Kalman filters), no learning, no functional restoration |
| Mechanistic interpretability | Observational only, no intervention capability |

### 3.2 The Core Novelty Claims

1. **The ANP framework** — formally defining recording and stimulation as dual problems
   under bandwidth-limited, biologically plausible constraints. This framing does not
   exist in the literature.
2. **PC as anomaly/drift detector** — using predictive coding's prediction error as a
   self-supervised signal for detecting representational drift in a source network
   through a compressed interface. To our knowledge, novel.
3. **Error column as a first-class representational structure** — standard PC treats
   errors as transient training signals. SP-CANN encodes them, compresses them into a
   shared latent space, and decodes them into an actionable stimulation signal. The
   error pathway is symmetric with the representation pathway.
4. **Functional restoration via learned intervention** — a small biologically plausible
   auxiliary network that statistically reduces performance degradation in a decaying
   source network. First systematic investigation of computational function replacement
   through bandwidth-limited neural interfaces.
5. **The artificial EEG dataset** — a reproducible, controlled proxy for biological
   neural interface bandwidth constraints. Novel in construction and useful to the
   broader community.

### 3.3 The One-Sentence Claim

> We are the first to formally investigate whether a biologically plausible,
> bandwidth-constrained auxiliary network can learn to detect and compensate for
> progressive functional degradation in a source network — a problem simultaneously
> relevant to neural prosthetics, continual learning, and mechanistic interpretability,
> that no existing framework addresses.

---

## 4. Research Questions

**RQ1 (Recording):** Can an auxiliary network, operating under biological plausibility
and bandwidth constraints, learn to decode meaningful information from a source
network's internal activity?

**RQ2 (Stimulation):** Can an auxiliary network intervene in a source network's
operation to restore or compensate for degraded functional capability?

These two questions are experimentally separable. RQ1 (detection + decoding) can be
fully investigated and published independently of RQ2 (stimulation + restoration).

---

## 5. Progression Ladder: MNIST-FFNN to SP-CANN

Each step changes exactly one thing. This is the experimental discipline that makes
each rung independently defensible.

| Step | Model | What changes | Status |
|------|-------|-------------|--------|
| 1 | FFNN supervised | Cornerstone. Backprop + CE. Validates infra. | Done ✓ |
| 2 | PC-FFNN | Replace backprop with PC local rules. | Done ✓ (97.3%) |
| 3 | PC-CNN | Add convolutional layers to PC model. | Done ✓ (99.1%) |
| 4 | Source + AUX split | Freeze source. Small PC-FFNN reads all activations. | Next |
| 5 | PC-AUX + population interface | Impose bandwidth constraint. Sweep C. Info-limit curve. | Next |
| 6a | PC-AUX encoder/decoder | Representation column only. Validate latent space. | PC-EncDec done (97.1%) |
| 6b | + Error column | Add ε encoder, shared z, decode to stimulation. | Novel contribution |
| 7 | SNN-AUX generative | Replace dense AUX neurons with LIF. Separate encoder/PC weights. | Planned |
| 8 | + Decay simulation | Progressive neuron ablation in source. Measure ΔF detection. | Planned |
| 9 | + Stimulation loop | Inject `s_out = −α·ε_dec` at interface. 3-condition trial. | Planned |
| 10 | SP-CANN | Full model. No backprop. Local rules only. | Target |

**Critical design note for step 7:** the SNN encoder and the PC generative model must
have *separate weight matrices*. Sharing weights causes catastrophic failure (confirmed
empirically in anp_spcnn iters 2–5 — all variants collapsed to chance). The SNN
encoder reads the population signal and produces spike trains. The PC model operates
on those spike trains with its own independent weights.

---

## 6. Sprint Plan and Publication Map

### Phase 1 — Infrastructure (months 1–6)

- **S1:** Source network baselines (FFNN, CNN, SNN) — done
- **S2:** Procedural GoL dataset
- **S3:** Artificial EEG dataset and population interface
- → **P2:** Artificial EEG paper (dataset + population encoding mechanism)
- → **P3:** GoL dataset paper

### Phase 2 — PC Auxiliary Baseline (months 4–10)

- **S4:** PC generative model of normal activity (step 6a)
- **S5:** Error column + shared latent space (step 6b)
- → **P1:** BPML review (written in parallel throughout)

### Phase 3 — Recording / RQ1 (months 8–18)

- **S6:** Bandwidth constraint — information limit curve (C vs decoding accuracy)
- **S7:** Decay detection — ΔF as drift signal, layer-wise localisation
- **S8:** Decoding accuracy — task readout from `r_out`
- **S9:** Ablation studies — channel count, placement, decay severity grid
- **S10:** SNN auxiliary implementation (step 7)
- → **P4:** Recording paper (full RQ1 result)

### Phase 4 — Stimulation / RQ2 (months 16–36)

- **S11:** Decay simulation protocol — progressive ablation, controlled severity levels
- **S12:** Stimulation signal — closed-loop `s_out` injection, gain α sweep
- **S13:** Three-condition trial — intact / decay / restored across 5+ seeds
- **S14:** Transfer and second-order learning extensions
- → **P5:** Restoration paper (core RQ2 result — the headline contribution)
- → **P6:** Transfer/meta paper (if time permits)

### Critical Path

S4 → S7 → S13. Everything else is parallel or extensional. If S13 produces a
statistically significant result, the PhD is complete. If it produces partial results,
the PhD is still complete with an honest characterisation of limits.

---

## 7. Ablation Grid (The Core Experiment)

The primary empirical output is this table, filled in across 5+ seeds:

| Channels C | Decay d | acc_intact | acc_decay | acc_restored | ΔF |
|------------|---------|------------|-----------|--------------|-----|
| 16 | 0% | — | — | — | — |
| 64 | 0% | — | — | — | — |
| 128 | 0% | — | — | — | — |
| 128 | 10% | — | — | — | — |
| 128 | 20% | — | — | — | — |
| 128 | 30% | — | — | — | — |
| 128 | 40% | — | — | — | — |

Each row is a parallel PBS job on CHPC. The autoresearch framework handles scheduling,
result extraction, and leaderboard updates autonomously.

---

## 8. What Already Exists (Leaderboard Summary)

| Model | Val acc | Relevance |
|-------|---------|-----------|
| CNN (vanilla) | 99.17% ± 0.10% | Source network baseline |
| PC-CNN v2 | 99.11% ± 0.08% | Best PC result — confirms PC scales to conv |
| SNN-CNN (rate, T=25) | 98.87% ± 0.13% | Confirms LIF neurons viable for spatial tasks |
| PC-EncDec v2 @ 60ep | 97.14% ± 0.21% | Step 6a already explored |
| PC-FFNN v3 | 97.30% ± 0.22% | PC local rules confirmed working |
| SPC-FFNN (all variants) | ~11% (chance) | Shared SNN/PC weights = fatal. Lesson learnt. |

The SPC-FFNN failure (iters 2–5) is not a dead end — it is a documented, understood
failure mode that directly informs the SP-CANN design (separate weight matrices).

---

## 9. Key Design Decisions to Discuss with Supervisor

1. **Source network choice:** should the source network also be a PC model, or a
   standard CNN/SNN? A standard CNN is a cleaner surrogate for a biological system.
   A PC source creates interesting interactions with a PC auxiliary but complicates
   the experimental isolation.

2. **Decay simulation mechanism:** neuron ablation (zero weights), synaptic weight
   noise, or gradual weight decay? Ablation is cleanest for experimental control.
   Weight noise is more biologically realistic. Need to justify choice relative to
   AD pathology (predominantly synaptic loss before cell death).

3. **Online vs offline auxiliary training:** does the auxiliary train before decay
   is introduced (learning normal, then tested against decay), or does it train
   continually during decay? The former maps to RQ1 cleanly; the latter introduces
   continual learning dynamics that are interesting but harder to control.

4. **Gain α in stimulation:** fixed, learned, or adaptive? A fixed α is a clean
   first result. A learned α is a more powerful but harder-to-train second result.
   Suggest fixed first.

5. **Biological plausibility spectrum:** the thesis explicitly operates on a spectrum
   from fully plausible (LIF, STDP, no backprop) to pragmatic relaxation (convolutional
   weight sharing, surrogate gradients). The relaxations are justified by compute
   constraints and do not undermine the CS claims. This should be stated explicitly
   in the thesis as a design axis rather than a limitation.

6. **Publication 4 framing:** the "meta-learning" paper is better framed as
   *second-order learning* — the auxiliary network learning a model of the source
   network's learning dynamics, not just its activity. This avoids confusion with
   the MAML/Prototypical Networks literature and connects more naturally to both
   RQ1 and RQ2.

---

## 10. Inspirations and Connections to Literature

| Inspiration | What it contributes to SP-CANN |
|-------------|-------------------------------|
| Predictive coding (Rao & Ballard 1999; Friston) | Core learning mechanism. Local error signals. No backprop. |
| VAE / autoencoder | Shared latent space z. Encoder/decoder structure. |
| U-Net (Ronneberger 2015) | Skip connections preserving fine-grained interface signal. |
| GLOM (Hinton 2021) | Lateral/recurrent within-level consensus. Column structure. |
| Forward-Forward (Hinton 2022) | Alternative to backprop. Goodness-based local learning. |
| SNN / snntorch | LIF neuron implementation. Rate and TTFS encoding. |
| Neural prosthetics (DBS, cochlear) | Application domain. Interface-as-stimulation analogy. |
| LoRA / adapter networks | Foil for novelty argument. SP-CANN is the bandwidth-constrained, biologically plausible, online alternative. |

---

## 11. Training Loop, Datasets, and Tasks

This is the most operationally complex part of the research because there are three
distinct learning loops running at different timescales, and the auxiliary network
has no direct gradient signal from the source.

### 11.1 The Three Loops

```
Loop 1 — Source network training (offline, once)
  ├── Train source network on task dataset to convergence
  ├── Freeze all source weights
  └── Source is never updated again

Loop 2 — Auxiliary pre-training (offline, before decay)
  ├── Run source network on task dataset (inference only)
  ├── Record population signal e_t at each timestep
  ├── Train SP-CANN auxiliary on {e_t} to minimise F
  ├── No labels needed — purely self-supervised
  └── Convergence criterion: F stabilises on held-out normal activity

Loop 3 — Auxiliary online loop (during decay evaluation)
  ├── For each decay step d:
  │   ├── Apply decay to source network (ablate d% of neurons)
  │   ├── Run source + auxiliary on evaluation dataset
  │   ├── Auxiliary reads e_t, computes ΔF (detection)
  │   ├── [RQ2 only] Auxiliary injects s_out = −α·ε_dec
  │   ├── Record: task accuracy, ΔF, layer-wise ε^l
  │   └── [Optional] Auxiliary continues updating on shifted e_t
  └── Repeat across decay levels d ∈ {0, 10, 20, 30, 40, 50}%
```

**Key point:** Loop 2 and Loop 3 are deliberately separated. The auxiliary learns
what *normal* looks like in Loop 2, then is tested against abnormal activity in Loop 3.
This clean separation is what makes the anomaly detection claim falsifiable — the
auxiliary has never seen decayed activity during training.

### 11.2 The PC Weight Update Rule (No Backprop)

During Loop 2, the auxiliary updates its weights using local PC inference. At each
layer `l`, the weight update is:

```
ΔW^l ∝ ε^l · (r^l)ᵀ
```

This is a Hebbian-like rule: strengthen connections that reduce prediction error.
No gradient flows between layers — each layer only needs its own error signal `ε^l`
and its own representation `r^l`. This is what makes the learning rule local and
biologically plausible.

For the SNN variant (step 7+), the representation `r^l` is replaced by spike train
statistics (rate-coded: mean firing rate over window T), and the weight update becomes:

```
ΔW^l ∝ ε^l · s̄^{l⊤},   s̄^l = (1/T) Σ_{t=1}^T s^l_t
```

### 11.3 Inference Phase (PC Settling)

Before each weight update, the PC model runs an inference phase: representations
`r^l` are iteratively updated to minimise `F` with weights held fixed:

```
r^l ← r^l − λ · ∂F/∂r^l,   for K steps
```

This is the fundamental PC dynamic. Weights update slowly (learning); representations
update fast (inference). `K` is a hyperparameter — typically 10–20 steps. This is
the same mechanism that produced the slow convergence observed in PC-EncDec experiments
(iters 7–10), and the same reason flat learning rate outperformed cosine decay:
aggressive LR decay stalls inference-phase credit assignment before full convergence.

### 11.4 Datasets and Task Progression

| Dataset | Type | Source network | Why |
|---------|------|----------------|-----|
| MNIST | Static image, classification | CNN / PC-CNN | Clean baseline. Already have all results. Easy to measure decay. |
| N-MNIST | Event-based, classification | SNN-CNN | Neuromorphic version of MNIST. Natural fit for spiking source. |
| Procedural GoL | Spatiotemporal, prediction | SNN / PC-RNN | Interpretable dynamics. Decay has visible, traceable effects. |
| Split-MNIST | Sequential classification | Any | Standard continual learning benchmark. Tests cross-task decay. |

**Recommended progression:**

1. Start with **MNIST + CNN source.** This is the cleanest possible setup — you have
   the source network trained, you understand its failure modes under ablation, and
   classification accuracy is an unambiguous metric. This is where the anomaly
   detection claim (RQ1) is first validated.
2. Move to **N-MNIST + SNN source** once the dense version works. This is the
   biologically motivated version — event-based input, spiking source, spiking auxiliary.
3. Use **GoL** for the spatiotemporal dynamics paper (P3) and as a secondary evaluation
   dataset for the restoration claim. GoL's interpretable structure makes it easier
   to *visualise* what the auxiliary is learning.
4. **Split-MNIST** is a stretch goal for the continual learning angle.

### 11.5 Decay Simulation Protocol

Three candidate mechanisms, in order of experimental cleanliness:

**Mechanism A — Hard ablation (recommended first):**

Randomly zero out `d%` of neurons (set all incoming and outgoing weights to zero).
Instantaneous, reproducible, seed-controlled. Worst-case for the auxiliary — abrupt
rather than gradual. Clean for isolating the restoration claim.

```
a^l_decayed = m ⊙ a^l,   m_i ~ Bernoulli(1−d),   m fixed per seed
```

**Mechanism B — Weight noise (more biologically realistic):**

Add Gaussian noise to synaptic weights proportional to `d`. Models synaptic
degradation rather than cell death.

```
W^l_decayed = W^l + η · N(0, d · |W^l|_F)
```

**Mechanism C — Progressive weight decay (most biologically realistic):**

Multiply weights by a decay factor at each timestep, simulating tau protein
accumulation. Produces gradual rather than step-change degradation.

```
W^l_t = W^l_{t-1} · (1 − γ_d)^t
```

Starting with Mechanism A gives the conservative result; if it works, the claim is
strong. Mechanisms B and C are natural follow-ups.

### 11.6 Evaluation Metrics

| Metric | What it measures | When used |
|--------|-----------------|-----------|
| Task accuracy (acc) | Source network performance on classification | All conditions |
| ΔF | Magnitude of energy increase post-decay | RQ1 detection |
| Layer-wise `\|ε^l\|` | Localisation of decay within source network | RQ1 localisation |
| `I(e; y)` (mutual information) | How much class information survives compression | Population interface |
| Acc recovery ratio ρ | `(acc_restored − acc_decay) / (acc_intact − acc_decay)` | RQ2 restoration |
| Auxiliary parameter count | Verifies auxiliary is genuinely small | All |
| Inference latency | Verifies auxiliary is feasible for real-time use | Final SP-CANN |

The **recovery ratio ρ** normalises the restoration result against the size of the
damage, giving a single number in [0, 1] where ρ = 1 means full restoration and
ρ = 0 means no improvement. This is more interpretable than raw accuracy, especially
when comparing across decay levels.

### 11.7 What a Single Training Run Looks Like End-to-End

```
1. Load pre-trained source network (e.g. CNN, MNIST, 99.17%)
2. Run Loop 2:
   - Feed 60,000 MNIST training samples through frozen source
   - Record e_t = W_pop · a_t + noise for each sample
   - Train SP-CANN auxiliary for N epochs on {e_t}
   - Validate: F should be low and stable on held-out normal e
3. Evaluate baseline (d = 0%):
   - Run source + auxiliary on 10,000 MNIST test samples
   - Record acc_intact, F_normal
4. For each decay level d ∈ {10, 20, 30, 40, 50}%:
   a. Apply ablation mask to source (fixed per seed)
   b. [Condition 1] Run source alone → record acc_decay
   c. [Condition 2] Run source + auxiliary (detection only) → record ΔF
   d. [Condition 3] Run source + auxiliary (stimulation on) → record acc_restored
5. Repeat steps 1–4 for 5 seeds
6. Compute ρ per (seed, d), report mean ± std
```

---

## 12. Open Questions for Supervisor

1. What is the preferred decay simulation mechanism — hard ablation first, or weight
   noise to be more biologically motivated?
2. Should the source network be a standard CNN or a PC model? A standard CNN gives
   cleaner experimental isolation.
3. How many inference steps K is reasonable for the PC settling phase, given real-time
   constraints?
4. Should the auxiliary be permitted to continue updating online during Loop 3 (decay
   evaluation), or is a frozen auxiliary the cleaner first result?
5. Is Publication 4 (second-order learning) worth pursuing independently, or should it
   be a section within Publication 5?
6. What is the target venue for Publication 5? Nature Machine Intelligence, Neural
   Networks, NeurIPS, or ICLR?
7. Is three years sufficient for S1–S13, or should S14 and P6 be explicitly scoped out?
8. Does the supervisor have existing CHPC allocation that can be shared, or should a
   new allocation be applied for?
