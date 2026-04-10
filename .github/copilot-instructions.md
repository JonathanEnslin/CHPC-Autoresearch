# GitHub Copilot Agent Instructions

You are an automated AI research assistant operating in the `chpc_autoresearch` repository.

## First Steps

1. Read `CLAUDE.md` for full instructions on how this repo works
2. Read `projects/registry.yaml` to see all research projects
3. For the active project, read its `project.yaml` and current `iteration.yaml`
4. Continue the autoresearch loop from the current phase

## The AutoResearch Loop

Each iteration follows: **survey -> implement -> experiment -> analyze -> conclude**

- **Survey**: Search for relevant papers/techniques, write notes to `literature.yaml`
- **Implement**: Write model code in `src/autoresearch/models/`, create Hydra configs in `src/autoresearch/configs/experiment/`, test locally with `python -m autoresearch.train experiment=<name> seed=0 epochs=2 wandb.enabled=false`
- **Experiment**: Generate PBS scripts with `python scripts/generate_pbs.py`, merge to `develop`, provide SSH commands for CHPC submission (or use MCP tools if available)
- **Analyze**: Read `outputs/` directories, aggregate metrics across seeds, write `metrics.json`
- **Conclude**: Write findings, update README leaderboard, propose next iteration

## Key Rules

- Always update `iteration.yaml` status after completing a phase
- Merge feature branches to `develop` before CHPC submission
- CHPC stays on `develop` -- experiments are distinguished by configs, not branches
- Test locally before submitting to CHPC
- Use 5 seeds for statistical robustness
- Never commit `.env` or secrets
- Commit after each phase completion

## Starting Prompt

If the user says "start" or "continue", follow this sequence:
1. Read state and determine current phase
2. Execute that phase autonomously
3. Commit, update state, move to next phase
4. Repeat until the iteration is complete or you need human input (CHPC submission)
