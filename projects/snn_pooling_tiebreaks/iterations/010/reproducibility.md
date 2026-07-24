# CIFAR-10 deterministic pooling reference

This is the reproducibility record for the completed five-seed reference in
iteration 10. Raw checkpoints and per-epoch metrics remain on CHPC under
`outputs/train/cifar10_snn_pooling_vgg9_tet_fashion_dynamics_post_deterministic_5seed_wandb_offline/`.

## Result

Final validation accuracy was **88.87% ± 0.72%** (sample standard deviation,
five seeds):

| Seed | Final validation accuracy | Final validation loss |
|---:|---:|---:|
| 0 | 89.06% | 0.561386 |
| 1 | 87.88% | 0.598404 |
| 2 | 88.58% | 0.547667 |
| 3 | 89.84% | 0.520529 |
| 4 | 88.98% | 0.556938 |

The mean final firing rates were 12.54% at stage-1 conv-1, 9.07%, 8.54%, and
5.02% after the three pooling sites, 1.15% in stage-4 conv-2, and 11.02% at
the spiking output. Thus the result is not explained by deep or output-layer
silence.

## Exact experiment preset

Hydra preset:
`cifar10_snn_pooling_vgg9_tet_fashion_dynamics_post_deterministic`

- Dataset: CIFAR-10, 10% validation split; training augmentation is horizontal
  flip (p=0.5), 32-pixel random crop with 4-pixel reflect padding, followed by
  normalization with mean `(0.4914, 0.4822, 0.4465)` and standard deviation
  `(0.2470, 0.2435, 0.2616)`.
- Model: six-timestep, direct-input, fully spiking VGG-9 (`[64, 128, 256,
  512]` channels; two convolutions per stage); 2x2 post-LIF deterministic max
  pooling after stages 1--3; spiking output and TET decoding.
- Neuron dynamics: LIF beta `0.5`, threshold `1.0`, ATan surrogate with slope
  `25`, immediate reset (`reset_delay=false`), no BNTT.
- Loss: temporal efficient cross entropy with temporal weight `0.1`.
- Optimizer: AdamW, LR `0.02`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight
  decay `1e-4`; global gradient-norm clipping `5.0`.
- Schedule: 200 epochs, cosine annealing from `0.02` to `0.0002`; batch size
  128; four data-loader workers; early stopping disabled.
- Seeds: `0, 1, 2, 3, 4`.

The first sequential PBS job reached its walltime and the first continuation
exposed a PyTorch 2.6 checkpoint-loading compatibility issue. The final result
uses the exact fixed recipe above: seed 3 resumed from epoch 28 after changing
the trusted local checkpoint load to `weights_only=False`; no completed seed
was rerun.
