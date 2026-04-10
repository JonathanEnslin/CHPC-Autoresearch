# CHPC Survival Guide

> Source: https://wiki.chpc.ac.za/survival_guide
> Status: Placeholder -- run `/docs-fetch` to pull full content

## Quick Reference

- **Login**: `ssh username@lengau.chpc.ac.za`
- **Internet node**: `ssh chpclic1` (from login node, for git/pip)
- **Lustre storage**: `cd lustre/` or `/mnt/lustre/users/{username}/`
- **Submit job**: `qsub script.pbs`
- **Check jobs**: `qstat -u username`
- **Cancel job**: `qdel job_id`
- **Module list**: `module avail`
- **Load Python**: `module load chpc/python/anaconda/3-2024.10.1`
