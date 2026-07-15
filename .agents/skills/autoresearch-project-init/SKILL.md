---
name: autoresearch-project-init
description: Create a new structured AutoResearch project and first iteration. Use when asked to begin a new research topic or initialize a project under projects/.
---

Obtain a human-readable name, safe project slug, description, and initial research direction when they are not supplied.

Create `projects/<slug>/project.yaml` with active status, iteration 1, the stated direction, and an empty best result. Create `projects/<slug>/iterations/001/iteration.yaml` with status `planned`, the goal, date, phase history, outcomes, and human-feedback fields. Add the project to `projects/registry.yaml`.

Create the appropriate feature branch before implementation work. Keep CHPC on `develop`; experiments are separated by Hydra configuration and output directory, not CHPC branches.
