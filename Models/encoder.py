import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import DepthwiseSeparableConv
from .dropout import Dropout
from .Normalizations import get_norm
from .Activations import get_activation


def mask_logits(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mask: True means masked (PAD) positions.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return target.masked_fill(mask, -1e30)


class PosEncoder(nn.Module):
    """
    Sinusoidal positional encoding as a non-trainable buffer.
    x: [B, C, L]
    """
    def __init__(self, d_model: int, length: int):
        super().__init__()
        freqs = torch.tensor(
            [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)],
            dtype=torch.float32
        ).unsqueeze(1)  # [C, 1]
        phases = torch.tensor(
            [0.0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)],
            dtype=torch.float32
        ).unsqueeze(1)
        pos = torch.arange(length, dtype=torch.float32).repeat(d_model, 1)
        pe = torch.sin(pos * freqs + phases)  # [C, L]
        self.register_buffer("pos_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        return x + self.pos_encoding[:, :length]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.drop = Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L], mask: [B, L] True=PAD
        batch_size, channels, length = x.size()
        x = x.transpose(1, 2)  # [B, L, C]

        q = self.q_linear(x).view(batch_size, length, self.num_heads, self.d_k)
        k = self.k_linear(x).view(batch_size, length, self.num_heads, self.d_k)
        v = self.v_linear(x).view(batch_size, length, self.num_heads, self.d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)

        if mask.dtype != torch.bool:
            mask = mask.bool()
        attn_mask = mask.unsqueeze(1).expand(-1, length, -1).repeat(self.num_heads, 1, 1)  # [B*h, L, L]

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = mask_logits(attn, attn_mask)
        attn = F.softmax(attn, dim=2)
        attn = self.drop(attn)

        out = torch.bmm(attn, v)  # [B*h, L, d_k]
        out = out.view(self.num_heads, batch_size, length, self.d_k)
        out = out.permute(1, 2, 0, 3).contiguous().view(batch_size, length, self.d_model)
        out = self.fc(out)
        out = self.drop(out)
        return out.transpose(1, 2)  # [B, C, L]


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, conv_num: int, k: int, length: int, init_name: str = "kaiming", act_name: str = "relu", norm_name: str = "layer_norm", norm_groups: int = 8):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, k, init_name=init_name) for _ in range(conv_num)])
        self.self_att = MultiHeadAttention(d_model, num_heads, dropout)
        self.fc1 = nn.Linear(d_model, d_model, bias=True)
        self.fc2 = nn.Linear(d_model, d_model, bias=True)
        self.pos = PosEncoder(d_model, length)
        self.act = get_activation(act_name)
        self.dropout = dropout
        self.conv_num = conv_num

        # [OLD] Normalization over [C, L]; fixed length required for layer_norm.
        # [FIX] Normalization: LayerNorm over C (channel-only); GroupNorm over [C/G, L].
        # normb → conv[0]; norms[0..conv_num-2] → conv[1..conv_num-1]; norms[conv_num-1] → self-attn; norme → FFN
        self.normb = get_norm(norm_name, d_model, length, num_groups=norm_groups)
        self.norms = nn.ModuleList([get_norm(norm_name, d_model, length, num_groups=norm_groups) for _ in range(conv_num)])
        self.norme = get_norm(norm_name, d_model, length, num_groups=norm_groups)

    # [OLD] element-wise Dropout only on even conv sublayers, no layer dropout
    # [FIX] Paper Section 4.1: stochastic depth — sublayer l survival prob p_l = 1-(l/L)(1-p_L)
    def _layer_dropout(self, inputs: torch.Tensor, residual: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if self.training and drop_prob > 0:
            if torch.empty(1).uniform_(0, 1).item() < drop_prob:
                return residual
            return F.dropout(inputs, p=drop_prob, training=True) + residual
        return inputs + residual

    def forward(self, x: torch.Tensor, mask: torch.Tensor, l: int = 1, total_layers: int = 0) -> torch.Tensor:
        # drop_rate per sublayer: dropout * l / L (0 when stochastic depth is disabled)
        drop_scale = self.dropout / total_layers if total_layers > 0 else 0.0
        out = self.pos(x)

        for i, conv in enumerate(self.convs):
            res = out
            out = self.normb(out) if i == 0 else self.norms[i - 1](out)
            if i % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = conv(out)
            out = self.act(out)
            out = self._layer_dropout(out, res, drop_scale * l)
            l += 1

        res = out
        out = self.norms[self.conv_num - 1](out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self._layer_dropout(out, res, drop_scale * l)
        l += 1

        res = out
        out = self.norme(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = out.transpose(1, 2)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = out.transpose(1, 2)
        out = self._layer_dropout(out, res, drop_scale * l)

        return out
