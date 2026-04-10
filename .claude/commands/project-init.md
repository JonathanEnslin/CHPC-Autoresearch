# Project Init

Create a new research project.

## Steps

1. Ask the user for:
   - Project name (human-readable)
   - Project slug (kebab-case, e.g., `image_processing_nn`)
   - Description
   - Initial direction/goal

2. Create project directory: `projects/{slug}/`

3. Create `projects/{slug}/project.yaml`:
   ```yaml
   name: "Project Name"
   slug: project_slug
   status: active
   created: "YYYY-MM-DD"
   description: "..."
   current_iteration: 1
   branch: "project/{slug}"
   human_direction: "..."
   best_result:
     model: null
     dataset: null
     accuracy: null
     experiment_config: null
   ```

4. Create first iteration: `projects/{slug}/iterations/001/iteration.yaml`:
   ```yaml
   project: project_slug
   iteration: 1
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

5. Add entry to `projects/registry.yaml`

6. Create a feature branch for the first iteration: `git checkout -b feature/{slug}-iter-1-setup`
7. All development work happens on feature branches, merged to `develop` when ready
8. CHPC always stays on `develop` -- experiments are distinguished by configs, not branches
