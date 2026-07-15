---
name: autoresearch-iteration-init
description: Start the next structured iteration in an existing AutoResearch project. Use when asked to advance a project, create an iteration, or set a new iteration goal.
---

Read the project's `project.yaml`, the previous iteration outcomes and findings, and any human direction or feedback. Propose a goal based on that evidence and obtain confirmation if it has not been supplied.

Increment `current_iteration`, create zero-padded `projects/<slug>/iterations/NNN/iteration.yaml` with status `planned`, and update the project and registry state. Create a feature branch for the iteration before implementation work.
