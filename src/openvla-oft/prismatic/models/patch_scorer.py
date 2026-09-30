"""
patch_scorer.py

Lightweight bidirectional Transformer that assigns a logit to each vision patch (VLA-ACL).
"""

import torch
import torch.nn as nn


class PatchScorer(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 5,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Identity() if vision_dim == hidden_dim else nn.Linear(vision_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim), enable_nested_tensor=False
        )
        self.logit_head = nn.Linear(hidden_dim, 1)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """Maps vision patch embeddings (B, N, vision_dim) to patch logits (B, N)."""
        return self.logit_head(self.encoder(self.input_proj(vision_tokens))).squeeze(-1)


def select_top_k_per_view(patch_logits: torch.Tensor, k: int, num_views: int) -> torch.Tensor:
    """Returns sorted indices (B, num_views * k) of the `k` highest-scoring patches in each camera view (Eq. 4.2)."""
    batch_size, num_patches = patch_logits.shape
    patches_per_view = num_patches // num_views
    view_top_k = patch_logits.reshape(batch_size, num_views, patches_per_view).topk(k, dim=-1).indices
    view_offsets = torch.arange(num_views, device=patch_logits.device).view(1, num_views, 1) * patches_per_view
    return (view_top_k + view_offsets).reshape(batch_size, -1).sort(dim=-1).values
