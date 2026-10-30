import copy
import warnings
from typing import Sequence, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from detco.tapbackbone import TapBackbone  # Wrapper that extracts intermediate feature maps
from lightly.models.modules.heads import MoCoProjectionHead


class Yolo_Detco(nn.Module):
    """
    DetCo-style pretraining on a YOLO backbone.

    The model creates two projection head pyramids per stage –
    a *global* head and a *local* head – for each tapped feature map.
    If local patches are concatenated along the channel dimension,
    local heads will have an increased input dimensionality.

    Args:
        backbone_seq (nn.Sequential): The feature extractor without final head.
        head_layer_ids (Sequence[int], optional): Indices in `backbone_seq` after
            which feature maps are extracted. Defaults to (2, 4, 6, 9).
        patch_count (int): Number of patches per image for local pathway.
            Used to scale the input channels of the local heads.
        hidden_dim (int): Hidden dimension of each MLP head.
        out_dim (int): Output embedding size.
        normalize (bool): Whether to apply L2 normalization in the heads.
    """
    def __init__(
        self,
        backbone_seq: nn.Sequential,
        head_layer_ids: Optional[Sequence[int]] = None,
        *,
        patch_count: int = 9,
        hidden_dim: int = 2048,
        out_dim: int = 128,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if head_layer_ids and max(head_layer_ids) != len(backbone_seq) - 1:
            warnings.warn("⚠️  Final head is not placed on the last backbone layer!", stacklevel=2)
        if head_layer_ids is None:
            head_layer_ids = (2, 4, 6, 9)

        # ---------------- Backbones -----------------------------------
        # Main and momentum encoder (copied without gradient)
        self.q_backbone = TapBackbone(backbone_seq, tap_ids=tuple(head_layer_ids))
        self.k_backbone = copy.deepcopy(self.q_backbone)

        # Channel dimensions at each tap point (e.g., [32, 64, 128, 256])
        channels: List[int] = self.q_backbone.channels
        self.n_stages = len(channels)
        self.patch_count = patch_count

        # TODO Normalisierung der Feature-Maps?
        # ---------------- Projection Heads ----------------------------
        # Each head block: Adaptive Pool → Flatten → ProjectionHead
        self.q_global_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Global average pool: (B, C, H, W) → (B, C, 1, 1)
                nn.Flatten(1),                 # Flatten: (B, C, 1, 1) → (B, C)
                MoCoProjectionHead(            # Final head
                    input_dim=c,
                    hidden_dim=hidden_dim,
                    output_dim=out_dim
                )
            )
            for c in channels
        ])
        self.k_global_heads = copy.deepcopy(self.q_global_heads)

        self.q_local_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # (B*P, C, H, W) → (B*P, C, 1, 1)
                nn.Flatten(1),                 # → (B*P, C)
                MoCoProjectionHead(
                    input_dim=c * patch_count,  # Channels multiplied by number of patches
                    hidden_dim=hidden_dim,
                    output_dim=out_dim
                )
            )
            for c in channels
        ])
        self.k_local_heads = copy.deepcopy(self.q_local_heads)

        # ---------------- Freeze momentum encoder ---------------------
        for p in self.k_backbone.parameters():
            p.requires_grad = False
        for p in self.k_global_heads.parameters():
            p.requires_grad = False
        for p in self.k_local_heads.parameters():
            p.requires_grad = False

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _encode_global(self, backbone: TapBackbone, heads: nn.ModuleList, x: torch.Tensor):
        """
        Compute global embeddings from tapped feature maps.
        Args:
            backbone: The feature extractor (query or momentum).
            heads: The corresponding global projection heads.
            x: Input tensor (B, 3, H, W)
        Returns:
            List[Tensor]: Global embeddings at each tap stage.
        """
        feats = backbone(x)
        return [head(f) for head, f in zip(heads, feats)]

    def _encode_local(self, backbone: TapBackbone, heads: nn.ModuleList, x_patches: torch.Tensor):
        """
        Compute local embeddings from image patches.

        Args:
            backbone: Feature extractor (query or momentum).
            heads: Local projection heads.
            x_patches: Tensor of shape (B*P, 3, H, W)

        Returns:
            List[Tensor]: Local embeddings at each tap stage.
        """
        feats = backbone(x_patches)  # (B*P, C, H, W)
        outs: List[torch.Tensor] = []
        P = self.patch_count
        B = x_patches.size(0) // P

        for f_map, head in zip(feats, heads):
            _, C, H, W = f_map.shape
            f_map = f_map.reshape(B, P * C, H, W)  # Concatenate patches along channel axis
            outs.append(head(f_map))              # → (B, out_dim)
        return outs

    # -----------------------------------------------------------------
    def forward(
        self,
        g1: torch.Tensor,
        g2: torch.Tensor,
        l1: torch.Tensor,
        l2: torch.Tensor,
    ):
        """
        Forward pass: computes query/key embeddings for global and local pathways.

        Args:
            g1, g2: Global images (B, 3, H, W)
            l1, l2: Local patch views (B*P, 3, H, W)

        Returns:
            Dict[str, List[Tensor]]: Projection outputs for all four branches.
        """
        q_global = self._encode_global(self.q_backbone, self.q_global_heads, g1)
        k_global = self._encode_global(self.k_backbone, self.k_global_heads, g2)

        q_local  = self._encode_local(self.q_backbone, self.q_local_heads, l1)
        k_local  = self._encode_local(self.k_backbone, self.k_local_heads, l2)

        return {
            "q_global": q_global,
            "k_global": k_global,
            "q_local": q_local,
            "k_local": k_local,
        }

    # -----------------------------------------------------------------
    @torch.no_grad()
    def update_momentum(self, m: float = 0.99):
        """
        Exponential Moving Average (EMA) update for the momentum encoder.

        Args:
            m (float): Momentum factor, typically 0.99 or 0.999
        """
        # Backbone update
        for p_q, p_k in zip(self.q_backbone.parameters(), self.k_backbone.parameters()):
            p_k.data.mul_(m).add_(p_q.data, alpha=1 - m)

        # Global head update
        for q_mod, k_mod in zip(self.q_global_heads, self.k_global_heads):
            for p_q, p_k in zip(q_mod.parameters(), k_mod.parameters()):
                p_k.data.mul_(m).add_(p_q.data, alpha=1 - m)

        # Local head update
        for q_mod, k_mod in zip(self.q_local_heads, self.k_local_heads):
            for p_q, p_k in zip(q_mod.parameters(), k_mod.parameters()):
                p_k.data.mul_(m).add_(p_q.data, alpha=1 - m)
