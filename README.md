# Action Consistency Learning for Visual Token Pruning in VLA Models
Model checkpoints available shortly. Find the paper manuscript [here](main.pdf).

VLA-ACL learns which vision tokens a Vision-Language-Action model needs. A lightweight patch scorer is trained on top of
a frozen, fine-tuned [OpenVLA-OFT](https://github.com/moojink/openvla-oft) policy so that the actions predicted from the
top-K vision patches per camera view stay consistent with the full-context actions and the ground-truth actions. At
inference time, only the top-K patches are passed to the LLM backbone.

All code lives in [`src/openvla-oft`](src/openvla-oft). See [SETUP.md](src/openvla-oft/SETUP.md) for installation and
[LIBERO.md](src/openvla-oft/LIBERO.md) for the LIBERO setup and datasets. Run the commands below from `src/openvla-oft`.

## Training a Patch Scorer

```bash
python vla-scripts/train_scorer.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-10 \
  --data_root_dir modified_libero_rlds \
  --dataset_name libero_10_no_noops \
  --soft_top_k 64
```

## Evaluating with Vision Patch Pruning

Use a locally stored OpenVLA-OFT checkpoint, so that its modeling code is synced with this repo.

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --patch_scorer.enabled True \
  --patch_scorer.checkpoint runs/<RUN_ID>/patch_scorer_head--5000_checkpoint.pt \
  --patch_scorer.prune_topk 64
```

## Acknowledgements

This codebase builds on [OpenVLA-OFT](https://github.com/moojink/openvla-oft).
