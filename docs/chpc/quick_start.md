# CHPC Quick Start

> Source: https://wiki.chpc.ac.za/quick:start
> Status: Placeholder -- run `/docs-fetch` to pull full content

## PBS Queue Options

| Queue | Resources | Use Case |
|-------|-----------|----------|
| `serial` | CPU only, up to 24 cores | Data preprocessing, CPU training |
| `smp` | Single node, large memory | Large batch training |
| `normal` | Multi-node CPU | Distributed CPU jobs |
| `gpu_1` | 1 GPU + CPUs | Single GPU training |
| `gpu_2` | 2 GPUs + CPUs | Multi-GPU training |
| `gpu_3` | 3 GPUs + CPUs | Multi-GPU training |
| `gpu_4` | 4 GPUs + CPUs | Multi-GPU training |

## PBS Resource Syntax

```
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=12:00:00
```

## Common Commands

```bash
qsub script.pbs          # Submit a job
qstat -u $USER           # Check your jobs
qdel <job_id>            # Cancel a job
module avail             # List available modules
module load <module>     # Load a module
```
