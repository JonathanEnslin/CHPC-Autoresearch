# State Resume

Read the current state and determine what to do next. Use this at the start of a new session.

## Steps

1. Read `projects/registry.yaml`
2. For each active project:
   - Read `projects/{slug}/project.yaml`
   - Read current iteration: `projects/{slug}/iterations/{NNN}/iteration.yaml`
   - Report: project name, iteration number, current phase/status, goal
3. Determine the next action based on the iteration's `status`:
   - `planned` -> Begin survey phase
   - `survey` -> Continue or complete literature survey
   - `implement` -> Continue or complete implementation
   - `experiment` -> Check experiment status, submit if needed
   - `analyze` -> Analyse results
   - `conclude` -> Write conclusions and findings
   - `completed` -> Start next iteration or await direction
4. Check for any `human_feedback` or `human_direction` that guides the work
5. Summarize the current state to the user and propose next steps
