import torch
import torch.nn as nn

from .layernorm import LayerNorm
from .groupnorm import GroupNorm


class _ChannelFirstLayerNorm(nn.Module):
    """LayerNorm wrapper for channel-first [B, C, L] tensors.

    Transposes to [B, L, C], applies LayerNorm(d_model) over the channel
    dimension only (standard Transformer convention), then transposes back.
    Parameters shape is [d_model], independent of sequence length.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


# Registry: norm_name -> class
normalizations = {
    "layer_norm": LayerNorm,
    "group_norm":  GroupNorm,
}


def get_norm(name: str, d_model: int, length: int = 0, num_groups: int = 8) -> nn.Module:
    """
    Instantiate a normalization module by registry name.

    Args:
        name:       one of "layer_norm", "group_norm"
        d_model:    number of channels (C)
        length:     (unused, kept for API compatibility)
        num_groups: number of groups; used only by group_norm

    Returns:
        nn.Module instance of the requested normalization.

    Shapes:
        "layer_norm" → _ChannelFirstLayerNorm(d_model)
            normalizes over channel dim of [B, d_model, length]
            parameters shape: [d_model] (length-independent)
        "group_norm"  → GroupNorm(num_groups, d_model)
            normalizes over [C/G, *spatial] per group of [B, d_model, *]
    """
    if name not in normalizations:
        raise ValueError(
            f"Unknown normalization '{name}'. Available: {list(normalizations.keys())}"
        )
    if name == "layer_norm":
        return _ChannelFirstLayerNorm(d_model)
    else:  # group_norm
        return GroupNorm(num_groups, d_model)
