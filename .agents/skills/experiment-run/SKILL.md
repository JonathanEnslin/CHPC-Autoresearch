---
name: experiment-run
description: Run or validate an AutoResearch experiment locally, or prepare it for CHPC. Use when asked to execute, smoke-test, or sweep an experiment configuration.
---

List relevant presets under `src/autoresearch/configs/experiment/` when the configuration is unclear.

Use `python -m autoresearch.train experiment=<name> seed=0 epochs=2 wandb.enabled=false` for a quick local validation. Use a single seed for a full local run or `--multirun experiment=<name>` for the configured multi-seed sweep. Pass Hydra overrides directly on the command line.

Record meaningful local or completed results in the current iteration's `experiments.yaml`. Use `$chpc-submit` for CHPC submission.
