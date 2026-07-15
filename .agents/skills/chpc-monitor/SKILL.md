---
name: chpc-monitor
description: Monitor submitted CHPC jobs, process completed experiment results, and prepare a Codex Desktop scheduled-task prompt. Use when asked to poll, monitor, or automatically follow CHPC experiments.
---

Perform one immediate status check using `$chpc-status`.

For each verified completed experiment, read remote per-seed `status.json` and logs, calculate mean and standard deviation across completed seeds, write `metrics.json`, update `experiments.yaml` and `iteration.yaml`, update conclusions and the README leaderboard as appropriate, and commit only the tracked state and documentation files. Raw `outputs/` remain on CHPC because they are gitignored.

For jobs that died before all required seeds completed, diagnose the log and resubmit only after correcting the cause or increasing walltime. Training resumes from `last_checkpoint.pt`.

When recurring monitoring is requested in Codex Desktop, create or guide the user to create a scheduled task with this workflow and the requested interval. In a surface without scheduled tasks, provide the reusable polling prompt instead of claiming a background loop exists.
