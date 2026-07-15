---
name: chpc-submit
description: Generate, validate, submit, and track a CHPC PBS experiment job for this AutoResearch repository. Use when asked to run an experiment on CHPC or submit a PBS job.
---

Determine the experiment configuration and confirm any material ambiguity. First run a local two-epoch smoke test with W&B disabled unless a relevant validation already exists.

Generate the PBS file with `scripts/generate_pbs.py`, using an explicit job name and `python -m autoresearch.train --multirun experiment=<config>`. PBS scripts may contain private CHPC details and are intentionally ignored by Git.

Ensure the required code is on `develop`. Read `.env` without printing secrets, connect to CHPC, pull `develop` in `$CHPC_LUSTRE_PATH/$CHPC_REPO_NAME`, and submit with `qsub`. Record the job name, ID, configuration, resources, and status in the current `experiments.yaml`.

If SSH or submission is unavailable, provide exact manual commands. Estimate walltime as seeds × per-seed runtime × 1.2; use a conservative 4h for RNN-class models and 2h for FFNN/CNN if no estimate exists.
