# AutoResearch Architecture

## Overview

AutoResearch is a framework for automated AI-driven research. It combines:

1. **Config-driven experimentation** (Hydra + PyTorch) for running ML experiments
2. **State tracking** (YAML files) so AI agents can resume across sessions
3. **CHPC integration** (PBS scripts) for HPC experiment execution
4. **Agent skills** (Claude Code commands) for specialized tasks

## The AutoResearch Loop

```
Human gives direction
        |
        v
  [1. Survey]     -- Literature review, gather references
        |
        v
  [2. Implement]  -- Write code, create configs, test locally
        |
        v
  [3. Experiment] -- Submit PBS jobs to CHPC, run multi-seed sweeps
        |
        v
  [4. Analyze]    -- Aggregate results, compute statistics
        |
        v
  [5. Conclude]   -- Write findings, update leaderboard
        |
        v
  Next iteration (or human redirects)
```

Each iteration is a complete cycle. State is persisted in YAML files so a fresh agent session can pick up at any phase.

## State Schema

### Project Registry (`projects/registry.yaml`)

Lists all projects with their slug, status, and current iteration number.

### Project (`projects/{slug}/project.yaml`)

Metadata about a research project: name, description, current iteration, human direction, best result achieved.

### Iteration (`projects/{slug}/iterations/{NNN}/iteration.yaml`)

Tracks the current phase of an iteration. The `status` field determines what the agent does next.

Valid status transitions: `planned` -> `survey` -> `implement` -> `experiment` -> `analyze` -> `conclude` -> `completed`

### Supporting Files (per iteration)

- `literature.yaml` -- References and notes from survey phase
- `experiments.yaml` -- Experiment runs, job IDs, results
- `metrics.json` -- Aggregated metrics from analysis
- `findings.md` -- Written conclusions

## Config System (Hydra)

Experiments are composed from modular YAML configs:

```
configs/
  train.yaml          <- base config (defaults, checkpoint, wandb, early stopping)
  dataset/            <- dataset configs (mnist, cifar10)
  model/              <- model configs (ffnn, cnn, resnet18)
  optimizer/          <- optimizer configs (adam, sgd)
  loss_fn/            <- loss function configs (cross_entropy, mse, bce)
  experiment/         <- preset compositions (mnist_ffnn_adam, etc.)
```

An experiment config overrides the base defaults:

```yaml
# @package _global_
defaults:
  - override /dataset: mnist
  - override /model: ffnn
  - override /optimizer: adam
  - override /loss_fn: cross_entropy
```

Any parameter can be overridden at the command line.

## CHPC Integration

1. PBS script template in `templates/experiment.pbs.template`
2. Generator script `scripts/generate_pbs.py` reads `.env` for CHPC credentials
3. Jobs are submitted via `qsub` after SSH-ing to Lengau
4. Results are retrieved via `git pull` or `scp`

## Directory Layout

```
src/autoresearch/           <- Python package
  train.py                  <- Main entry point (@hydra.main)
  configs/                  <- Hydra YAML configs
  models/                   <- FFNN, CNN, ResNet18
  datasets/                 <- Custom datasets (empty initially)
  utils/                    <- data, device, evaluation, reproducibility, wandb

projects/                   <- Research state (version controlled)
templates/                  <- PBS script templates
scripts/                    <- Helper scripts
docs/                       <- Documentation
outputs/                    <- Experiment outputs (gitignored)
data/                       <- Downloaded datasets (gitignored)
```
