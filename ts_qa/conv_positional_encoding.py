# Adapted from https://github.com/Jwoo5/fairseq-signals

import math

import torch.nn as nn
import torch

from .same_pad import SamePad

class ConvPositionalEncoding(nn.Module):
    def __init__(self, encoder_embed_dim: int, conv_pos: int, conv_pos_groups: int):
        super().__init__()

        self.embedding_dim = encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size = conv_pos,
            padding = conv_pos // 2,
            groups = conv_pos_groups
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean = 0, std = std)
        nn.init.constant_(self.pos_conv.bias, 0) # type: ignore

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name = "weight", dim = 2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())

    def forward(self, x, channel_first=False) -> torch.Tensor:
        if not channel_first:
            x = x.transpose(1,2)
        x_conv = self.pos_conv(x).transpose(1,2)

        return x_conv