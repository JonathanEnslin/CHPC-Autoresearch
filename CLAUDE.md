# AutoResearch - Agent Instructions

## Kickoff

Use the `/start` command to begin or resume the autoresearch loop:

```
/start [YOUR RESEARCH TOPIC]
```

This reads state, determines what to do next, and works through each phase autonomously. For CHPC experiments, the agent tries SSH directly. If SSH isn't available, it provides manual commands.

## What This Repo Is

An automated AI research assistant that runs ML experiments on South Africa's CHPC (Center for High Performance Computing). The agent follows the scientific method in iterative research loops, tracking state so new sessions can pick up where the last left off.

## Boot Sequence (New Session)

1. Read `projects/registry.yaml` to see all projects and their status
2. Find the active project, read its `projects/{slug}/project.yaml`
3. Read the current iteration: `projects/{slug}/iterations/{NNN}/iteration.yaml`
4. Check the `status` field to determine the current phase
5. Execute that phase (see Research Phases below)
6. If the human gave specific instructions in their prompt, follow those instead

## Research Phases

Each iteration follows this sequence:

| Phase | Status Value | What To Do |
|-------|-------------|------------|
| Plan | `planned` | Read the project goal and human direction. Plan what to do. |
| Survey | `survey` | Search for relevant papers/techniques. Write notes to `literature.yaml`. |
| Implement | `implement` | Write/modify code in `src/autoresearch/`. Create experiment configs. |
| Experiment | `experiment` | Generate PBS scripts, submit to CHPC, track job IDs in `experiments.yaml`. |
| Analyze | `analyze` | Read outputs, compute statistics across seeds, write `metrics.json`. |
| Conclude | `conclude` | Write findings, update project best_result, update README leaderboard. |
| Done | `completed` | Advance to next iteration or await human direction. |

After completing a phase, update `iteration.yaml` status to the next phase.

## Running Experiments

**Locally:**
```bash
python -m autoresearch.train experiment=mnist_ffnn_adam seed=0 epochs=2 wandb.enabled=false
```

**Multi-seed sweep:**
```bash
python -m autoresearch.train --multirun experiment=mnist_ffnn_adam
```

**On CHPC:** Use `/chpc-submit` skill (agent will try SSH directly).

**Override any config at CLI:**
```bash
python -m autoresearch.train experiment=mnist_ffnn_adam seed=42 batch_size=64 epochs=20
```

## CHPC SSH (Direct Agent Access)

The agent should attempt to SSH to CHPC directly. This enables fully autonomous experiment submission.

**How it works:**
1. Read `.env` for `CHPC_USERNAME`, `CHPC_HOST`, `CHPC_LUSTRE_PATH`, `CHPC_REPO_NAME`
2. SSH to CHPC: `ssh $CHPC_USERNAME@$CHPC_HOST "command"`
3. Sync code: `ssh $CHPC_USERNAME@$CHPC_HOST "cd $CHPC_LUSTRE_PATH/$CHPC_REPO_NAME && git pull origin develop"`
4. Submit job: `ssh $CHPC_USERNAME@$CHPC_HOST "cd $CHPC_LUSTRE_PATH/$CHPC_REPO_NAME && qsub experiments/<name>.pbs"`
5. Check status: `ssh $CHPC_USERNAME@$CHPC_HOST "qstat -u $CHPC_USERNAME"`

**Authentication:**
- **Local Claude Code**: Uses the user's existing SSH keys (~/.ssh/)
- **Remote GitHub Copilot**: Uses SSH key stored as a GitHub secret (write to ~/.ssh/id_rsa before use)

**Fallback:** If SSH fails (no keys, network issues), provide the user with the exact commands to run manually. Never get stuck -- always have a fallback.

## Key Paths

| Path | Purpose |
|------|---------|
| `src/autoresearch/` | Main Python package |
| `src/autoresearch/configs/` | Hydra YAML configs |
| `src/autoresearch/configs/experiment/` | Experiment presets |
| `src/autoresearch/models/` | Model implementations |
| `src/autoresearch/utils/` | Shared utilities |
| `projects/` | Research state tracking |
| `projects/registry.yaml` | Master project index |
| `templates/` | PBS script template |
| `scripts/generate_pbs.py` | PBS script generator |
| `docs/chpc/` | CHPC documentation (offline) |
| `outputs/` | Experiment outputs (gitignored) |
| `data/` | Downloaded datasets (gitignored) |

## Adding New Components

**New model:** Create `src/autoresearch/models/mymodel.py`, add config `src/autoresearch/configs/model/mymodel.yaml` with `_target_: autoresearch.models.mymodel.MyModel`, export in `models/__init__.py`.

**New dataset:** Add config `src/autoresearch/configs/dataset/mydataset.yaml` with `_target_:` pointing to a torchvision dataset or custom class.

**New experiment:** Create `src/autoresearch/configs/experiment/name.yaml` that overrides dataset, model, optimizer, loss_fn.

## Environment Variables (.env)

Copy `.env.example` to `.env` and fill in real values. Never commit `.env`.

| Variable | Purpose |
|----------|---------|
| `CHPC_USERNAME` | CHPC login username |
| `CHPC_PROJECT_ID` | PBS project code (e.g., CSCI1166) |
| `CHPC_EMAIL` | Email for PBS notifications |
| `CHPC_LUSTRE_PATH` | Lustre storage path |
| `CHPC_REPO_NAME` | Repo directory name on CHPC |
| `CHPC_MODULE_PYTHON` | Python module to load |
| `WANDB_API_KEY` | Weights & Biases API key |
| `WANDB_ENTITY` | W&B entity/team |
| `WANDB_PROJECT` | W&B project name |

## Git Workflow

- `main` -- stable releases
- `develop` -- integration branch where all code work happens

**Feature branches** (`feature/{slug}-iter-{N}-{desc}`) are used for development work (implementing new models, configs, utilities). Once the code is ready and tested locally, merge to `develop`.

**CHPC always stays on `develop`.** Experiments are distinguished by **Hydra configs**, not branches. This means multiple experiments can run concurrently on CHPC -- they use different output directories based on `experiment_group` and `seed`. Never checkout a feature branch on CHPC.

**Workflow:**
1. Agent creates a feature branch for implementation work
2. Agent writes code, configs, PBS scripts on the feature branch
3. Agent merges to `develop` when ready
4. Human (or CI) pulls `develop` on CHPC and submits PBS jobs
5. Multiple PBS jobs can run in parallel -- each writes to its own `outputs/` subdirectory

## Rules

- Never commit `.env` or any secrets
- Always update state files after completing a phase
- Test locally with `epochs=2 wandb.enabled=false` before submitting to CHPC
- Use 5 seeds for statistical robustness
- Keep experiment configs composable (dataset + model + optimizer + loss_fn)
- Outputs are gitignored; only state/configs/code are tracked
