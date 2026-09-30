"""
train_scorer.py

Trains a VLA-ACL patch scorer on top of a frozen, fine-tuned OpenVLA-OFT policy via action consistency learning.
Only the patch scorer is trained; the VLA, proprio projector, and action head stay frozen.

Usage:
    python vla-scripts/train_scorer.py \
        --vla_path moojink/openvla-7b-oft-finetuned-libero-10 \
        --data_root_dir modified_libero_rlds \
        --dataset_name libero_10_no_noops \
        --soft_top_k 64
"""

import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import draccus
import torch
import torch.nn as nn
import tqdm
from huggingface_hub import snapshot_download
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import wandb
from experiments.robot.openvla_utils import (
    apply_film_to_vla,
    check_model_logic_mismatch,
    find_checkpoint_file,
    load_component_state_dict,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.patch_scorer import PatchScorer, select_top_k_per_view
from prismatic.models.projectors import ProprioProjector
from prismatic.training.scorer_train_utils import (
    compute_scorer_training_loss,
    gate_vision_tokens,
    get_black_pixel_values,
    get_language_embeddings,
    get_vision_patch_embeddings,
    predict_actions,
    project_vision_patch_embeddings,
    soft_top_k,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TrainScorerConfig:
    # fmt: off
    vla_path: str = "moojink/openvla-7b-oft-finetuned-libero-10"  # Frozen fine-tuned OpenVLA-OFT policy (HF Hub or local)

    # Dataset
    data_root_dir: Path = Path("modified_libero_rlds")  # Directory containing RLDS datasets
    dataset_name: str = "libero_10_no_noops"            # Name of the dataset that `vla_path` was fine-tuned on
    run_root_dir: Path = Path("runs")                   # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 5_000                    # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Frozen VLA settings (MUST MATCH FINE-TUNING!)
    use_l1_regression: bool = True                   # Only continuous L1 regression action heads are supported
    num_images_in_input: int = 2                     # Number of images in the VLA input
    use_proprio: bool = True                         # If True, includes robot proprioceptive state in input
    use_film: bool = False                           # If True, the VLA uses FiLM to infuse language into visual features
    lora_rank: int = 32                              # (When `use_film==True`) Rank of LoRA weight matrix used in fine-tuning

    # Patch scorer architecture
    vision_scorer_hidden_dim: int = 1024             # Hidden dimension of the patch scorer Transformer
    vision_scorer_layers: int = 5                    # Number of patch scorer Transformer layers
    vision_scorer_heads: int = 16                    # Number of patch scorer attention heads

    # Action consistency loss (Eq. 4.8)
    gt_weight: float = 0.5                           # mu: weight of the L1 loss to ground-truth actions
    teacher_weight: float = 0.5                      # lambda: weight of the L1 loss to full-context teacher actions

    # Soft top-K selection
    soft_top_k: int = 64                             # K: number of vision tokens kept per camera view
    soft_top_k_tau: float = 0.1                      # Soft top-K alpha at the end of the cosine anneal (smaller => sharper)
    soft_top_k_max_tau: float = 2.0                  # Soft top-K alpha at the start of the cosine anneal (larger => softer)
    soft_top_k_end_step: Optional[int] = None        # Step at which alpha reaches `soft_top_k_tau` (None: `max_steps`)
    hard_top_k_metric: bool = True                   # If True, also logs the action loss under hard top-K pruning
    hard_top_k_log_freq: int = 50                    # Hard top-K logging frequency in steps (costs an extra LLM forward)

    # Training configuration
    batch_size: int = 8                              # Batch size
    learning_rate: float = 1e-4                      # Learning rate (cosine decay to 0 over `max_steps`)
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 5_000                           # Number of training steps
    save_freq: Optional[int] = None                  # Checkpoint saving frequency in steps (None: only at `max_steps`)
    image_aug: bool = True                           # If True, trains with image augmentations

    # Logging
    use_wandb: bool = True                           # If True, logs to Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def get_run_id(cfg: TrainScorerConfig) -> str:
    run_id = (
        f"vla-acl+{cfg.dataset_name}+k{cfg.soft_top_k}"
        f"+teacher-{cfg.teacher_weight}+gt-{cfg.gt_weight}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr-{cfg.learning_rate}"
    )
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def load_frozen_component(module: nn.Module, vla_path: str, name: str, device_id: int) -> nn.Module:
    """Loads the `{name}--{step}_checkpoint.pt` saved next to the fine-tuned VLA and freezes the module."""
    module.load_state_dict(load_component_state_dict(find_checkpoint_file(vla_path, name)))
    return module.to(device_id, dtype=torch.bfloat16).eval().requires_grad_(False)


def load_dataset_statistics(vla_path: str, dataset_name: str) -> Dict:
    """Loads the action/proprio statistics that the fine-tuned VLA was trained with."""
    with open(os.path.join(vla_path, "dataset_statistics.json")) as f:
        all_stats = json.load(f)
    if dataset_name in OXE_NAMED_MIXTURES:
        names = [name for name, _weight in OXE_NAMED_MIXTURES[dataset_name]]
    else:
        names = [dataset_name]
    return {name: all_stats[name] for name in names}


def get_selection_scores(scorer_logits: torch.Tensor, step: int, cfg: TrainScorerConfig) -> torch.Tensor:
    """Soft top-K selection scores (Eq. 2.7) computed per camera view, with alpha annealed by a cosine schedule."""
    anneal_progress = min(step, cfg.soft_top_k_end_step) / cfg.soft_top_k_end_step
    alpha_range = cfg.soft_top_k_max_tau - cfg.soft_top_k_tau
    alpha = cfg.soft_top_k_tau + 0.5 * alpha_range * (1 + math.cos(math.pi * anneal_progress))

    batch_size, num_patches = scorer_logits.shape
    per_view_logits = scorer_logits.float().reshape(batch_size * cfg.num_images_in_input, -1)
    return soft_top_k(per_view_logits, cfg.soft_top_k, alpha).reshape(batch_size, num_patches)


@torch.no_grad()
def compute_hard_top_k_metrics(
    vla: nn.Module,
    action_head: nn.Module,
    batch: Dict[str, torch.Tensor],
    device_id: int,
    projected: torch.Tensor,
    scorer_logits: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    teacher_actions: Optional[torch.Tensor],
    cfg: TrainScorerConfig,
) -> Dict[str, float]:
    """Action losses when keeping only the top-K vision tokens per view at their original positions, as at inference."""
    num_vision_patches = scorer_logits.shape[1]
    keep_indices = select_top_k_per_view(scorer_logits, cfg.soft_top_k, cfg.num_images_in_input)
    kept_vision = torch.gather(
        projected[:, :num_vision_patches], dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
    )
    pruned_projected = torch.cat((kept_vision, projected[:, num_vision_patches:]), dim=1)
    position_ids = OpenVLAForActionPrediction.build_preserving_multimodal_position_ids(
        batch_size=projected.shape[0],
        device=projected.device,
        vision_keep_indices=keep_indices,
        num_vision_patches=num_vision_patches,
        num_proprio_tokens=projected.shape[1] - num_vision_patches,
        num_tail_tokens=batch["input_ids"].shape[1] - 1,
    )
    hard_predicted_actions = predict_actions(vla, action_head, batch, device_id, pruned_projected, position_ids)

    hard_selection_scores = torch.zeros_like(scorer_logits).scatter_(1, keep_indices, 1.0)
    _, metrics = compute_scorer_training_loss(
        predicted_actions=hard_predicted_actions,
        ground_truth_actions=ground_truth_actions,
        teacher_actions=teacher_actions,
        selection_scores=hard_selection_scores,
        gt_weight=cfg.gt_weight,
        teacher_weight=cfg.teacher_weight,
        log_unmasked_gt_gap=False,
    )
    return metrics


def save_scorer_checkpoint(run_dir: Path, step: int, scorer_head: nn.Module) -> None:
    checkpoint_path = run_dir / f"patch_scorer_head--{step}_checkpoint.pt"
    torch.save(scorer_head.state_dict(), checkpoint_path)
    print(f"Saved patch scorer checkpoint: {checkpoint_path}")


@draccus.wrap()
def train_scorer(cfg: TrainScorerConfig) -> None:
    """Trains a patch scorer on a frozen OpenVLA-OFT policy with the action consistency loss."""
    assert cfg.use_l1_regression, "Patch scorer training only supports continuous L1 regression action heads!"
    if cfg.soft_top_k_end_step is None:
        cfg.soft_top_k_end_step = cfg.max_steps
    if cfg.save_freq is None:
        cfg.save_freq = cfg.max_steps
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Training patch scorer on frozen VLA `{cfg.vla_path}` with dataset `{cfg.dataset_name}`")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    device_id = 0
    torch.cuda.set_device(device_id)

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # Load frozen VLA (synced with the modeling code in this repo)
    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    update_auto_map(cfg.vla_path)
    check_model_logic_mismatch(cfg.vla_path)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, default=str, indent=2)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if cfg.use_film:
        vla = apply_film_to_vla(vla, cfg.vla_path, cfg.lora_rank)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla = vla.to(device_id).eval().requires_grad_(False)

    # Load frozen proprio projector and action head
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = load_frozen_component(
            ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM), cfg.vla_path, "proprio_projector", device_id
        )
    action_head = load_frozen_component(
        L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM),
        cfg.vla_path,
        "action_head",
        device_id,
    )

    # Create trainable patch scorer
    scorer_head = PatchScorer(
        vision_dim=vla.vision_backbone.embed_dim,
        hidden_dim=cfg.vision_scorer_hidden_dim,
        num_layers=cfg.vision_scorer_layers,
        num_heads=cfg.vision_scorer_heads,
    ).to(device_id, dtype=torch.bfloat16)
    print(f"# trainable patch scorer params: {sum(p.numel() for p in scorer_head.parameters()):,}")
    optimizer = AdamW(scorer_head.parameters(), lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps, eta_min=0.0)

    # Create dataset with the fine-tuned VLA's normalization statistics
    batch_transform = RLDSBatchTransform(
        ActionTokenizer(processor.tokenizer),
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=cfg.num_images_in_input > 1,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        precomputed_dataset_statistics=load_dataset_statistics(cfg.vla_path, cfg.dataset_name),
    )
    save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0)

    # Black-image embeddings b (Eq. 4.4); with FiLM they depend on the language input and are computed per batch
    black_pixel_values = get_black_pixel_values(
        processor.image_processor.apply_transform, vla.config.image_sizes[0], cfg.num_images_in_input
    ).to(device_id, dtype=torch.bfloat16)
    if not cfg.use_film:
        black_embeddings = get_vision_patch_embeddings(vla, black_pixel_values)

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    with tqdm.tqdm(total=cfg.max_steps, leave=True) as progress:
        scorer_head.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            ground_truth_actions = batch["actions"].to(device_id, dtype=torch.bfloat16)

            # Full-context vision patch embeddings X and teacher actions (Eq. 4.7)
            pixel_values = batch["pixel_values"].to(device_id, dtype=torch.bfloat16)
            language_embeddings = None
            if cfg.use_film:
                language_embeddings = get_language_embeddings(vla, batch, device_id)
                black_embeddings = get_vision_patch_embeddings(
                    vla, black_pixel_values.expand(pixel_values.shape[0], -1, -1, -1), language_embeddings
                )
            vision_tokens = get_vision_patch_embeddings(vla, pixel_values, language_embeddings)
            projected = project_vision_patch_embeddings(
                vla, vision_tokens, batch, device_id, cfg.use_proprio, proprio_projector
            )
            teacher_actions = None
            if cfg.teacher_weight > 0.0:
                with torch.no_grad():
                    teacher_actions = predict_actions(vla, action_head, batch, device_id, projected)

            # Soft-pruned student actions (Eqs. 4.5, 4.6)
            scorer_logits = scorer_head(vision_tokens)
            selection_scores = get_selection_scores(scorer_logits, gradient_step_idx, cfg)
            gated_vision_tokens = gate_vision_tokens(vision_tokens, selection_scores, black_embeddings)
            predicted_actions = predict_actions(
                vla,
                action_head,
                batch,
                device_id,
                project_vision_patch_embeddings(
                    vla, gated_vision_tokens, batch, device_id, cfg.use_proprio, proprio_projector
                ),
            )

            loss, metrics = compute_scorer_training_loss(
                predicted_actions=predicted_actions,
                ground_truth_actions=ground_truth_actions,
                teacher_actions=teacher_actions,
                selection_scores=selection_scores,
                gt_weight=cfg.gt_weight,
                teacher_weight=cfg.teacher_weight,
            )
            (loss / cfg.grad_accumulation_steps).backward()
            recent_losses.append(metrics["action_loss"])

            if (batch_idx + 1) % cfg.grad_accumulation_steps != 0:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.update()
            smoothed_loss = sum(recent_losses) / len(recent_losses)
            progress.set_postfix(loss=f"{smoothed_loss:.3f}", refresh=False)

            if cfg.use_wandb and gradient_step_idx % cfg.wandb_log_freq == 0:
                log_dict = {
                    "Scorer Train/Smoothed Loss": smoothed_loss,
                    "Scorer Train/Learning Rate": scheduler.get_last_lr()[0],
                    **{f"Scorer Train/{name}": value for name, value in metrics.items()},
                }
                if cfg.hard_top_k_metric and gradient_step_idx % cfg.hard_top_k_log_freq == 0:
                    hard_top_k_metrics = compute_hard_top_k_metrics(
                        vla,
                        action_head,
                        batch,
                        device_id,
                        projected,
                        scorer_logits,
                        ground_truth_actions,
                        teacher_actions,
                        cfg,
                    )
                    log_dict.update({f"Scorer Train/Hard Top-K/{name}": value for name, value in hard_top_k_metrics.items()})
                wandb.log(log_dict, step=gradient_step_idx)

            completed_steps = gradient_step_idx + 1
            if completed_steps % cfg.save_freq == 0 or completed_steps == cfg.max_steps:
                save_scorer_checkpoint(run_dir, completed_steps, scorer_head)
            if completed_steps == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    train_scorer()
