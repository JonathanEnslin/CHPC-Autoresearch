# Iteration Init

Start a new iteration within an existing project.

## Steps

1. Read `projects/{slug}/project.yaml` to get `current_iteration`
2. Increment: `new_iter = current_iteration + 1`
3. Create directory: `projects/{slug}/iterations/{NNN}/` (zero-padded to 3 digits)
4. Create `iteration.yaml`:
   ```yaml
   project: project_slug
   iteration: N
   status: planned
   goal: "..."
   created: "YYYY-MM-DD"
   phase_history: []
   outcomes:
     summary: null
     key_metrics: {}
     artifacts: []
   human_feedback: null
   ```
5. Update `project.yaml`: set `current_iteration` to new value
6. Update `projects/registry.yaml`
7. Create feature branch: `git checkout -b feature/{slug}-iter-{N}-{short_desc}`

## Determining the Goal

- Read previous iteration's `outcomes` and `findings.md` if they exist
- Read `human_feedback` from previous iteration if present
- Read `human_direction` from project.yaml
- Propose a goal based on the above
