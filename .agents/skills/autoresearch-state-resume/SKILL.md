---
name: autoresearch-state-resume
description: Inspect AutoResearch project and iteration state, report the next research action, and resume the appropriate phase. Use at the start of a task or when asked for research-project status.
---

Read `projects/registry.yaml`. For every active project, read `project.yaml` and its current `iterations/NNN/iteration.yaml`.

Report the project, iteration, status, goal, human guidance, and the next action. Map statuses as follows: `planned` → survey, `survey` → complete survey, `implement` → implement, `experiment` → inspect or submit jobs, `analyze` → analyze results, `conclude` → conclude, and `completed` → await direction or propose the next iteration.

Do not alter state merely by reporting it.
