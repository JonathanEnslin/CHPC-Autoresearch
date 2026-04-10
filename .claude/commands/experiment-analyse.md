# Experiment Analyse

Analyse experiment results and produce insights.

## Steps

1. **Locate outputs**: Check `outputs/train/{experiment_group}/{seed}/` for each seed
2. **Read status files**: Each run has a `status.json` with completion status and best_val_loss
3. **Read configs**: Each run has a `config.yaml` with the full resolved configuration
4. **Aggregate metrics across seeds**:
   - Mean and std of accuracy and loss
   - Best seed and its metrics
   - Parameter count (from config or model)
5. **Compare experiments**: Create comparison tables across different model/dataset combinations
6. **Update state files**:
   - Write metrics to `projects/{slug}/iterations/{NNN}/metrics.json`
   - Update `experiments.yaml` with results
7. **Update iteration status** to `analyze` if not already

## Metrics to Collect

| Metric | Source |
|--------|--------|
| Best val loss | `status.json` -> `best_val_loss` |
| Final epoch | `status.json` -> `epoch` |
| Completion status | `status.json` -> `status` |
| Parameter count | From model config or `config.yaml` |
| Training time | From log timestamps |

## Output Format

Write `metrics.json`:
```json
{
  "experiment_group": "mnist_ffnn_adam",
  "seeds": [0, 1, 2, 3, 4],
  "mean_accuracy": 97.8,
  "std_accuracy": 0.15,
  "mean_loss": 0.071,
  "best_seed": 3,
  "best_accuracy": 98.01,
  "num_params": 109386
}
```
