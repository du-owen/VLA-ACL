

# VLA-ACL: Action-Consistent Visual Token Pruning for Efficient Vision-Language-Action Models

[![arXiv](https://img.shields.io/badge/arXiv-2610.08133-b31b1b.svg)](https://arxiv.org/abs/2610.08133)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/owendudu/VLA-ACL)

VLA-ACL learns which vision tokens a Vision-Language-Action model needs. A lightweight patch scorer is trained on top of
a frozen, fine-tuned [OpenVLA-OFT](https://github.com/moojink/openvla-oft) policy so that the actions predicted from the
top-K vision patches per camera view stay consistent with the full-context actions and the ground-truth actions. At
inference time, only the top-K patches are passed to the LLM backbone.

https://github.com/user-attachments/assets/cfcf3f39-19e7-4f3a-a720-8e662b2b76c5

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

## Downloading the Patch Scorers

Released scorers are on [owendudu/VLA-ACL](https://huggingface.co/owendudu/VLA-ACL): one file per LIBERO suite (`spatial`, `object`, `goal`, `10`) at `k32` and `k64`. Each file is a scorer `state_dict` (~130 MB). The eval defaults already match this architecture (5 layers, hidden size 1024, 16 heads).

```bash
huggingface-cli download owendudu/VLA-ACL --local-dir checkpoints/vla-acl
```

To fetch a single scorer:

```bash
huggingface-cli download owendudu/VLA-ACL scorer-libero-10-k64.pt --local-dir checkpoints/vla-acl
```

Use the scorer trained on the same suite as the OpenVLA-OFT checkpoint (`moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}`), and set `--patch_scorer.prune_topk` to the `K` in the filename (`32` or `64`).

## Evaluating with Vision Patch Pruning

Use a locally stored OpenVLA-OFT checkpoint, so that its modeling code is synced with this repo. The example below keeps 64 patches per camera view on LIBERO-10:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --patch_scorer.enabled True \
  --patch_scorer.checkpoint checkpoints/vla-acl/scorer-libero-10-k64.pt \
  --patch_scorer.prune_topk 64
```

A checkpoint from [Training a Patch Scorer](#training-a-patch-scorer) is loaded the same way. Pass its path as `--patch_scorer.checkpoint` and set `--patch_scorer.prune_topk` to the `K` used in that run.

## Citation

```bibtex
@misc{du2026vlaaclactionconsistentvisualtoken,
      title={VLA-ACL: Action-Consistent Visual Token Pruning for Efficient Vision-Language-Action Models},
      author={Owen Du and Yang Yue and Jie Zhang and Jiaqi Pi and Chi Bene Chen and Gao Huang},
      year={2026},
      eprint={2610.08133},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.08133},
}
```

## Acknowledgements

This codebase builds on [OpenVLA-OFT](https://github.com/moojink/openvla-oft).
