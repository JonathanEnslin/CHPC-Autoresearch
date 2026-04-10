# CHPC GPU Guide

> Source: https://wiki.chpc.ac.za/guide:gpu
> Status: Placeholder -- run `/docs-fetch` to pull full content

## GPU Queues on Lengau

GPU jobs use the `gpu_N` queues where N is the number of GPUs requested.

### PBS Example (Single GPU)

```bash
#PBS -q gpu_1
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
```

### PBS Example (Multi-GPU)

```bash
#PBS -q gpu_4
#PBS -l select=1:ncpus=16:mem=256gb:ngpus=4
```

## CUDA Setup

GPUs are available via CUDA after loading the appropriate module:

```bash
module load chpc/cuda/11.8
```

PyTorch with CUDA support is typically available through the Anaconda module.

## Tips

- Always request appropriate memory per GPU (at least 16GB per GPU)
- Use `nvidia-smi` in your job script to verify GPU availability
- Set `CUDA_VISIBLE_DEVICES` if you need specific GPU selection
