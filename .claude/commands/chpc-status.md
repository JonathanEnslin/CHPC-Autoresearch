# CHPC Status

Check the status of CHPC jobs.

## Steps

1. Provide the user with the command to check job status:
   ```bash
   ssh $CHPC_USERNAME@lengau.chpc.ac.za "qstat -u $CHPC_USERNAME"
   ```
2. If the user has the output, parse it to identify:
   - Job ID, name, status (Q=queued, R=running, C=completed, E=exiting)
   - Queue, walltime used
3. If jobs are complete, check log files:
   ```bash
   ssh $CHPC_USERNAME@lengau.chpc.ac.za "tail -50 lustre/$CHPC_REPO_NAME/logs/<job_name>.out"
   ssh $CHPC_USERNAME@lengau.chpc.ac.za "cat lustre/$CHPC_REPO_NAME/logs/<job_name>.err"
   ```
4. Update the iteration's `experiments.yaml` with status
