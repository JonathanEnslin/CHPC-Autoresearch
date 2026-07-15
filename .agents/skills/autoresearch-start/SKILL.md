---
name: autoresearch-start
description: Start or resume this repository's complete CHPC AutoResearch loop. Use when asked to begin research, resume an iteration, determine the next phase, or continue autonomous research work.
---

Read `AGENTS.md`, `projects/registry.yaml`, and each active project's current `iteration.yaml`.

For an active iteration, execute the phase indicated by `status`: survey, implement, experiment, analyze, or conclude. Update the iteration state after each completed phase. Respect `human_feedback` and `human_direction`.

If no suitable project exists, ask for a research topic and use `$autoresearch-project-init`.

For experiments, validate locally before CHPC submission, generate PBS scripts with `scripts/generate_pbs.py`, and use `$chpc-submit`. If CHPC access is unavailable, give exact manual commands rather than stopping.

At an iteration boundary, summarize findings and propose the next goal; obtain confirmation before creating the next iteration.
