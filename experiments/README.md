# experiments/

This directory holds PBS job scripts for submitting experiments to CHPC (Centre for High Performance Computing).

PBS scripts are **gitignored** (`*.pbs`) because they contain personal credentials. Generate them with:

```bash
python scripts/generate_pbs.py
```

## Example PBS Script

```sh
#!/bin/sh
#PBS -N my_experiment_name
#PBS -q gpu_1
#PBS -P <PROJECT_CODE>
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -o /mnt/lustre/users/<USERNAME>/<REPO_NAME>/logs/my_experiment_name.out
#PBS -e /mnt/lustre/users/<USERNAME>/<REPO_NAME>/logs/my_experiment_name.err
#PBS -m abe -M <YOUR_EMAIL>

set -euo pipefail

module purge
module load chpc/python/anaconda/3-2024.10.1

cd /mnt/lustre/users/<USERNAME>/<REPO_NAME>
source .venv/bin/activate

echo "=========================================="
echo "Job: my_experiment_name"
echo "=========================================="
echo "Start time: $(date)"
echo ""

python -m autoresearch.train --multirun experiment=my_experiment_config

echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
```

## Placeholders

| Placeholder | Description | Where to set |
|-------------|-------------|--------------|
| `<PROJECT_CODE>` | PBS project allocation code | `.env` → `CHPC_PROJECT_ID` |
| `<USERNAME>` | CHPC username | `.env` → `CHPC_USERNAME` |
| `<REPO_NAME>` | Repository directory name on Lustre | `.env` → `CHPC_REPO_NAME` |
| `<YOUR_EMAIL>` | Email for PBS job notifications | `.env` → `CHPC_EMAIL` |

Copy `.env.example` to `.env` and fill in these values before generating PBS scripts.

## Submitting Jobs

```bash
# Pull latest develop on CHPC, then:
qsub experiments/my_experiment_name.pbs

# Check job status
qstat -u <USERNAME>
```
