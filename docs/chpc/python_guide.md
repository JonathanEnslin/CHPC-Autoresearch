# CHPC Python Guide

> Source: https://wiki.chpc.ac.za/guide:python
> Status: Placeholder -- run `/docs-fetch` to pull full content

## Loading Python

```bash
module load chpc/python/anaconda/3-2024.10.1
```

## Virtual Environments

Always use a virtual environment for project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Installing Packages

From the internet node (`chpclic1`):

```bash
ssh chpclic1
cd lustre/your_project
source .venv/bin/activate
pip install -r requirements.txt
```

## Tips

- Install packages from the internet node (`chpclic1`), not login or compute nodes
- Use `pip install --no-cache-dir` if disk space is limited
- Pin package versions in `requirements.txt` for reproducibility
- The Anaconda module includes NumPy, SciPy, and other scientific packages
