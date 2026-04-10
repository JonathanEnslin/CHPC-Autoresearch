# CHPC Setup

Set up or sync the repository on CHPC.

## First Time Setup

Provide these commands for the user to run:

```bash
# 1. SSH to login node
ssh $CHPC_USERNAME@lengau.chpc.ac.za

# 2. SSH to internet node (for git/pip access)
ssh chpclic1

# 3. Navigate to lustre
cd lustre/

# 4. Clone the repo
git clone <repo_url> $CHPC_REPO_NAME
cd $CHPC_REPO_NAME

# 5. Load Python and create venv
module load chpc/python/anaconda/3-2024.10.1
python -m venv .venv
source .venv/bin/activate

# 6. Install dependencies
pip install -e .

# 7. Create logs directory
mkdir -p logs

# 8. Copy .env.example to .env and fill in values
cp .env.example .env
# Edit .env with real values
```

## Sync (After Changes)

```bash
ssh $CHPC_USERNAME@lengau.chpc.ac.za
ssh chpclic1
cd lustre/$CHPC_REPO_NAME
git pull
source .venv/bin/activate
pip install -e .
```
