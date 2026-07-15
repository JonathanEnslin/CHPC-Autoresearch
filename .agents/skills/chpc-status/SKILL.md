---
name: chpc-status
description: Check CHPC PBS job status and update AutoResearch experiment tracking. Use when asked about queued, running, failed, or completed CHPC jobs.
---

Read `.env` without revealing secrets. Query `qstat -u <CHPC_USERNAME>` over SSH and interpret job ID, job name, state, queue, and elapsed walltime.

For jobs that have left the queue, inspect the matching `logs/<job_name>.out` and `.err` files and the per-seed output status files. Update the current iteration's `experiments.yaml` with accurate job status. Do not mark a job complete merely because it no longer appears in `qstat`; verify its output.

If CHPC access is unavailable, provide the exact check commands for the user to run and parse their pasted output.
