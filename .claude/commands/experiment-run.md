# Experiment Run

Run an experiment locally or prepare it for CHPC.

## Local Run (Quick Test)

```bash
python -m autoresearch.train experiment=<name> seed=0 epochs=2 wandb.enabled=false
```

## Local Run (Full)

```bash
python -m autoresearch.train experiment=<name> seed=0
```

## Multi-Seed Sweep

```bash
python -m autoresearch.train --multirun experiment=<name>
```

## Available Experiments

Check `src/autoresearch/configs/experiment/` for available presets:
- `mnist_ffnn_adam` -- FFNN on MNIST
- `mnist_cnn_adam` -- CNN on MNIST  
- `cifar10_resnet18_adam` -- ResNet-18 on CIFAR-10

## Override Parameters

Any config parameter can be overridden at CLI:
```bash
python -m autoresearch.train experiment=mnist_ffnn_adam seed=42 batch_size=64 epochs=100
```

## After Running

Update the iteration's `experiments.yaml` with results:
- Experiment name, config, seeds used
- Status (completed/failed)
- Key metrics (accuracy, loss)
- Output directory path
