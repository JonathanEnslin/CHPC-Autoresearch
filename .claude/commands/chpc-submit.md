# CHPC Submit

Generate a PBS script and submit it to CHPC.

## Steps

1. Determine which experiment config(s) to run (ask user if unclear)
2. Ensure current work is merged to `develop` and pushed
3. Generate the PBS script:
   ```bash
   python scripts/generate_pbs.py \
     --name <job_name> \
     --commands "python -m autoresearch.train --multirun experiment=<config_name>"
   ```
4. Commit the generated PBS script and push to `develop`
5. **Try SSH directly** to submit:
   ```bash
   # Read credentials from .env
   # Sync code on CHPC
   ssh $CHPC_USERNAME@$CHPC_HOST "cd $CHPC_LUSTRE_PATH/$CHPC_REPO_NAME && git pull origin develop"
   # Submit the job
   ssh $CHPC_USERNAME@$CHPC_HOST "cd $CHPC_LUSTRE_PATH/$CHPC_REPO_NAME && qsub experiments/<job_name>.pbs"
   ```
6. If SSH succeeds, capture the job ID from qsub output
7. Record the job name and ID in the current iteration's `experiments.yaml`
8. If SSH fails, provide the user with the exact commands to run manually

## Checking Status After Submission

```bash
ssh $CHPC_USERNAME@$CHPC_HOST "qstat -u $CHPC_USERNAME"
```

## Important: Branch Strategy
- **Merge to `develop` before submitting** -- CHPC always stays on `develop`
- Multiple experiments can run concurrently (each uses a different output directory)
- Never checkout feature branches on CHPC

## Notes
- The CHPC will email on job completion (configured in PBS -m abe -M)
- Use `--queue serial` for CPU-only jobs
- Override resources: `--queue gpu_1 --walltime 12:00:00 --ncpus 4 --mem 64gb --ngpus 1`
