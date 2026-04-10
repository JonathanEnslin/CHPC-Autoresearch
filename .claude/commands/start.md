# AutoResearch - Start / Resume

Read CLAUDE.md to understand this repository and how it works.

Then run /state-resume to understand the current state of all research projects.

Based on the current state, continue the autoresearch loop from wherever it left off:
- If no project exists yet, ask the user what research topic to pursue, then create one with /project-init.
- If a project exists with an active iteration, continue from the current phase.
- Work through each phase autonomously: survey -> implement -> experiment -> analyze -> conclude.
- For the experiment phase, generate PBS scripts. Try SSH to CHPC directly (see CHPC SSH section in CLAUDE.md). If SSH fails, provide the user with commands to run manually.
- After each phase, commit your work and update the state files before moving to the next phase.
- When an iteration completes, summarize findings and update the README leaderboard.
- Propose the next iteration's goal based on findings, then ask the user for confirmation before starting it.

$ARGUMENTS
