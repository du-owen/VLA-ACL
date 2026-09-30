"""
scorer_train_utils.py

Utilities for training the VLA-ACL patch scorer on a frozen OpenVLA-OFT model: soft top-K selection, black-image
gating, and forward passes of the frozen VLA on (soft-)pruned vision tokens.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint

from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

# Action components of LIBERO actions (dx, dy, dz, d_roll, d_pitch, d_yaw, gripper), for per-component metrics
ACTION_COMPONENT_SLICES: Dict[str, slice] = {
    "position": slice(0, 3),
    "orientation": slice(3, 6),
    "gripper": slice(6, 7),
}


@torch.no_grad()
def get_language_embeddings(vla: nn.Module, batch: Dict[str, torch.Tensor], device_id: int) -> torch.Tensor:
    """Returns input embeddings without the action tokens, as used for FiLM conditioning in fine-tuning."""
    input_embeddings = vla.get_input_embeddings()(batch["input_ids"].to(device_id))
    all_actions_mask = vla._process_action_masks(batch["labels"].to(device_id))
    return input_embeddings[~all_actions_mask].reshape(input_embeddings.shape[0], -1, input_embeddings.shape[2])


@torch.no_grad()
def get_vision_patch_embeddings(
    vla: nn.Module, pixel_values: torch.Tensor, language_embeddings: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Returns vision backbone patch embeddings (before LLM projection), FiLM-conditioned if given language embeddings."""
    return vla._preprocess_vision_features(
        pixel_values, language_embeddings=language_embeddings, use_film=language_embeddings is not None
    )


def get_black_pixel_values(
    image_transform: Callable[[Image.Image], torch.Tensor], resolution: int, num_images: int
) -> torch.Tensor:
    """Returns normalized pixel values (1, 6 * num_images, H, W) of an all-black image in every view (Eq. 4.4)."""
    return image_transform(Image.new("RGB", (resolution, resolution))).repeat(num_images, 1, 1).unsqueeze(0)


def project_vision_patch_embeddings(
    vla: nn.Module,
    vision_tokens: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    device_id: int,
    use_proprio: bool,
    proprio_projector: Optional[nn.Module],
) -> torch.Tensor:
    """Projects vision tokens into LLM space and appends the proprio token if used."""
    projected = vla._project_vision_features(vision_tokens)
    if use_proprio:
        proprio = batch["proprio"].to(device_id, dtype=projected.dtype)
        projected = vla._process_proprio_features(projected, proprio, proprio_projector)
    return projected


def gate_vision_tokens(
    vision_tokens: torch.Tensor, selection_scores: torch.Tensor, black_embeddings: torch.Tensor
) -> torch.Tensor:
    """Soft-prunes vision tokens toward the black-image embeddings: X~ = s * X + sqrt(1 - s^2) * b (Eq. 4.5)."""
    s = selection_scores.float().unsqueeze(-1)
    # Detached because d/ds sqrt(1 - s^2) diverges as s -> 1; gradients reach s through the s * X term
    black_weight = (1.0 - s.detach() ** 2).sqrt()
    gated = s * vision_tokens.float() + black_weight * black_embeddings.float()
    return gated.to(vision_tokens.dtype)


def _laplace_cdf(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0, 0.5 * torch.exp(x), 1.0 - 0.5 * torch.exp(-x))


@torch.no_grad()
def _solve_lapsum_threshold(scores: torch.Tensor, k: float, alpha: float, num_iters: int = 64) -> torch.Tensor:
    """Bisection for the threshold t satisfying sum_i Lap((scores_i - t) / alpha) = k (Eq. 2.7)."""
    lo = scores.min(dim=-1, keepdim=True).values - 10.0 * alpha
    hi = scores.max(dim=-1, keepdim=True).values + 10.0 * alpha
    for _ in range(num_iters):
        mid = 0.5 * (lo + hi)
        above_k = _laplace_cdf((scores - mid) / alpha).sum(dim=-1, keepdim=True) > k
        lo = torch.where(above_k, mid, lo)
        hi = torch.where(above_k, hi, mid)
    return 0.5 * (lo + hi)


class SoftTopK(torch.autograd.Function):
    """
    LapSum soft top-K (Struski et al., 2025). The threshold is found by bisection instead of LapSum's closed form
    (numerically identical); the backward pass is LapSum's closed-form vector-Jacobian product (Eq. A.20).
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: float, alpha: float) -> torch.Tensor:
        x = scores.float()
        threshold = _solve_lapsum_threshold(x, k, alpha)
        ctx.save_for_backward(x, threshold)
        ctx.alpha, ctx.dtype = alpha, scores.dtype
        return _laplace_cdf((x - threshold) / alpha).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        pdf = torch.exp(-(x - threshold).abs() / ctx.alpha) / (2.0 * ctx.alpha)
        grad_output = grad_output.float()
        weighted_mean = (pdf * grad_output).sum(dim=-1, keepdim=True) / pdf.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return (pdf * (grad_output - weighted_mean)).to(ctx.dtype), None, None


def soft_top_k(scores: torch.Tensor, k: int, alpha: float) -> torch.Tensor:
    """Maps scores (B, N) to selection scores in [0, 1] that sum to k along N; smaller alpha => sharper selection."""
    return SoftTopK.apply(scores, float(k), float(alpha))


def build_multimodal_embeddings(
    vla: nn.Module, batch: Dict[str, torch.Tensor], device_id: int, projected_patch_embeddings: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds [BOS, projected patches, prompt + (zeroed) action tokens] embeddings and attention mask."""
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    all_actions_mask = vla._process_action_masks(batch["labels"].to(device_id))

    input_embeddings = vla.get_input_embeddings()(input_ids) * ~all_actions_mask.unsqueeze(-1)
    projected_patch_embeddings = projected_patch_embeddings.to(dtype=input_embeddings.dtype)
    return vla._build_multimodal_attention(input_embeddings, projected_patch_embeddings, attention_mask)


def run_llm_backbone(
    vla: nn.Module,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Runs the LLM backbone (without LM head) and returns last hidden states, checkpointed when gradients flow."""
    llm = vla.language_model.model

    def _forward(embeddings, attention_mask, pos_ids):
        return llm(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=pos_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    if multimodal_embeddings.requires_grad:
        return checkpoint(_forward, multimodal_embeddings, multimodal_attention_mask, position_ids, use_reentrant=False)
    return _forward(multimodal_embeddings, multimodal_attention_mask, position_ids)


def predict_actions(
    vla: nn.Module,
    action_head: nn.Module,
    batch: Dict[str, torch.Tensor],
    device_id: int,
    projected_patch_embeddings: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Runs the frozen LLM + L1 regression action head on the given projected patches; returns (B, chunk, action_dim)."""
    multimodal_embeddings, multimodal_attention_mask = build_multimodal_embeddings(
        vla, batch, device_id, projected_patch_embeddings
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        last_hidden_states = run_llm_backbone(vla, multimodal_embeddings, multimodal_attention_mask, position_ids)

    # Same action hidden state indexing as `finetune.py`
    input_ids = batch["input_ids"].to(device_id)
    ground_truth_token_ids = batch["labels"].to(device_id)[:, 1:]
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    num_patches = last_hidden_states.shape[1] - input_ids.shape[1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(input_ids.shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    return action_head.predict_action(actions_hidden_states)


def _unweighted_component_l1(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Unweighted mean L1 per action component (position / orientation / gripper)."""
    err = (pred.float() - target.float()).abs()
    n = err.shape[-1]
    out: Dict[str, float] = {}
    for name, sl in ACTION_COMPONENT_SLICES.items():
        if sl.start >= n:
            continue
        end = min(sl.stop, n)
        if end > sl.start:
            out[name] = err[..., sl.start:end].mean().item()
    return out


def compute_scorer_training_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: Optional[torch.Tensor],
    teacher_actions: Optional[torch.Tensor],
    selection_scores: torch.Tensor,
    gt_weight: float = 0.5,
    teacher_weight: float = 0.5,
    log_unmasked_gt_gap: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Action consistency loss (Eq. 4.8): weighted L1 to the full-context teacher and to the ground-truth actions."""
    metrics: Dict[str, float] = {}

    action_loss = 0.0
    component_target = torch.zeros_like(predicted_actions)
    total_weight = gt_weight + teacher_weight

    if total_weight <= 0:
        raise ValueError("total_weight must be > 0")

    if gt_weight > 0.0:
        if ground_truth_actions is None:
            raise ValueError("gt_weight > 0 requires ground_truth_actions")
        gt_loss = (gt_weight / total_weight) * F.l1_loss(predicted_actions, ground_truth_actions)
        action_loss += gt_loss
        metrics["action_loss_gt"] = gt_loss.item()
        component_target += gt_weight * ground_truth_actions.float()
    if teacher_weight > 0.0:
        if teacher_actions is None:
            raise ValueError("teacher_weight > 0 requires teacher_actions")
        teacher_loss = (teacher_weight / total_weight) * F.l1_loss(predicted_actions, teacher_actions)
        action_loss += teacher_loss
        metrics["action_loss_teacher"] = teacher_loss.item()
        component_target += teacher_weight * teacher_actions.float()

    component_target /= total_weight

    metrics["action_loss"] = action_loss.item()
    metrics["loss"] = action_loss.item()
    metrics["scorer_mean"] = selection_scores.mean().item()

    with torch.no_grad():
        for name, value in _unweighted_component_l1(predicted_actions, component_target).items():
            metrics[f"action_loss_{name}"] = value
        if log_unmasked_gt_gap and teacher_actions is not None and ground_truth_actions is not None:
            metrics["action_loss_teacher_gt"] = F.l1_loss(teacher_actions, ground_truth_actions).item()

    return action_loss, metrics
