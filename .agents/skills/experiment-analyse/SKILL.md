---
name: experiment-analyse
description: Analyze AutoResearch experiment outputs across seeds and update research state. Use when asked to aggregate metrics, inspect completed runs, compare experiments, or write metrics.json.
---

Locate each seed under `outputs/train/<experiment_group>/<seed>/` or read the equivalent remote CHPC files. Verify completion from `status.json` before aggregating.

Compute mean and standard deviation for validation accuracy and loss, identify the best seed, record parameter count when available, and note incomplete or failed seeds separately. Write `projects/<slug>/iterations/<NNN>/metrics.json`, update `experiments.yaml`, and advance state only when the evidence supports it.

Compare against relevant baselines and update the README leaderboard during the conclude phase, not before.
