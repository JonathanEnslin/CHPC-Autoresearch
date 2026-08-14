# Minimal CHPC project kit

Copy this folder into an existing project when you need to run it on CHPC without bringing in the AutoResearch framework. It contains only operational instructions and a PBS template; it has no Python/package dependency on this repository.

## 1. Decide the remote project path

On your local machine, choose variables matching your CHPC account and allocation. Do not put passwords, API keys, or a copied `.env` in Git.

```bash
CHPC_USER="your_chpc_username"
CHPC_LOGIN="your_chpc_login_host"
CHPC_PROJECT="your_PBS_project_code"
REMOTE_ROOT="/mnt/lustre/users/$CHPC_USER"
REMOTE_PROJECT="$REMOTE_ROOT/my-project"
```

Use the normal CHPC login host for shell work, PBS submission, and monitoring:

```bash
ssh "$CHPC_USER@$CHPC_LOGIN"
mkdir -p "$REMOTE_PROJECT"
cd "$REMOTE_PROJECT"
qstat -u "$CHPC_USER"
```

## 2. Transfer code, data, and results through the SCP node

Use **`scp.chpc.ac.za`** for file transfers. The SCP node sees Lustre, so the destination can be the same absolute Lustre path used from the login and compute nodes.

```bash
# Upload a project directory (exclude large/generated content first if appropriate).
scp -r ./my-project "$CHPC_USER@scp.chpc.ac.za:$REMOTE_ROOT/"

# Upload a data archive, then unpack it from the normal login host.
scp ./data/dataset.tar.gz "$CHPC_USER@scp.chpc.ac.za:$REMOTE_PROJECT/data/"

# Download one output directory after a run.
scp -r "$CHPC_USER@scp.chpc.ac.za:$REMOTE_PROJECT/outputs/run-001" ./outputs/
```

Do **not** submit or monitor PBS jobs through the SCP node. Use the normal login host for `qsub`, `qstat`, and `qdel`.

For repeat code transfers, `rsync` over SSH is often more efficient if enabled for your account, but `scp -r` is the dependable baseline.

## 3. Reuse the existing SNN environment

An existing Lustre environment used by the SNN experiments is available at:

```bash
/mnt/lustre/users/jenslin/SNN/.snn-venv3
```

It has previously provided CUDA PyTorch, `snntorch`, Hydra/OmegaConf, and W&B. The AutoResearch checkout's `.venv` is a symlink to this environment; use the canonical path above in independent projects. Treat it as shared/known-good: activate and verify it, but do **not** install, upgrade, or remove packages in it.

```bash
source /mnt/lustre/users/jenslin/SNN/.snn-venv3/bin/activate
python - <<'PY'
import torch, snntorch
print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("snntorch:", snntorch.__version__)
PY
```

### Existing private environment file

The current private CHPC configuration file is on Lustre at:

```bash
/mnt/lustre/users/jenslin/chpc_autoresearch/.env
```

It contains connection/allocation values and W&B credentials used by the existing experiments. It is deliberately untracked. Do **not** download it, commit it, or transfer it to another system. If an existing project on CHPC needs those settings, it may load the private file *on CHPC only* immediately before starting its command:

```bash
set -a
source /mnt/lustre/users/jenslin/chpc_autoresearch/.env
set +a
```

Do not echo the values after loading them. New projects should keep their own project settings separate and only reuse this file where the existing private credentials are genuinely required.

If the project needs additional packages, create a project-local environment instead, preferably from an Internet-capable CHPC node such as `chpclic1` according to current CHPC policy:

```bash
cd "$REMOTE_PROJECT"
ssh chpclic1
cd "$REMOTE_PROJECT"
module load chpc/python/anaconda/3-2024.10.1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Do not assume compute nodes have Internet access. Install dependencies before submitting a GPU job, and pin versions in `requirements.txt` or a lockfile.

## 4. Submit a GPU job

Copy `PBS_TEMPLATE.pbs` into the project and replace every `CHANGE_ME` value. Set the project-specific command on the `python` line. Run from the normal login host:

```bash
cd "$REMOTE_PROJECT"
mkdir -p logs
qsub jobs/train_gpu.pbs
qstat -u "$USER"
```

Use a short smoke job before a long experiment. Then size walltime from an observed seed runtime:

```text
walltime = number of sequential seeds × observed time per seed × 1.2
```

The template writes a unique stdout/stderr pair beneath `logs/`; inspect these first whenever a job ends unexpectedly.

## 5. Monitor and recover

```bash
qstat -u "$USER"          # queued/running jobs
qdel <job_id>             # cancel a job
```

Inside the PBS job, `nvidia-smi` captures GPU/VRAM use in the log. Add checkpointing to the training script so a walltime-limited job can resume rather than restart. Keep code/configuration in Git and keep generated data, checkpoints, W&B credentials, and `.env` files out of Git.

## Minimal safe workflow

1. Develop and smoke-test locally.
2. Transfer code/data through `scp.chpc.ac.za`.
3. Log into the normal CHPC host, activate a verified environment, and run a tiny smoke PBS job.
4. Submit the real job from the login host; monitor with `qstat` and logs.
5. Download only the outputs you need through the SCP node.
