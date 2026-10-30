import torch
import torch.nn as nn
from typing import Iterable, Sequence, Tuple, List

class TapBackbone(nn.Module):
    """
    Lightweight wrapper that "taps" into feature maps at specified layer indices
    (`tap_ids`) and returns them as a list.

    The original backbone is *not* modified – only the `forward` pass is extended
    to collect the desired intermediate tensors.
    """
    def __init__(self, backbone: nn.Sequential, tap_ids: Iterable[int]):
        super().__init__()
        self.body = backbone
        self.tap_ids = set(tap_ids)  # use a set for O(1) lookups

        # --- Automatically infer the number of channels at tap points ---
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)  # dummy input
            channels: List[int] = []
            out = dummy
            for idx, layer in enumerate(self.body):
                out = layer(out)
                if idx in self.tap_ids:
                    channels.append(out.shape[1])
            self.channels: Tuple[int, ...] = tuple(channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass that collects feature maps at the specified tap layer indices.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            List[torch.Tensor]: List of tapped feature maps.
                Length equals `len(tap_ids)`.
        """
        feats: List[torch.Tensor] = []
        for idx, layer in enumerate(self.body):
            x = layer(x)
            if idx in self.tap_ids:
                feats.append(x)
        return feats  # [F_map_0, F_map_1, ...]