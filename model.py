import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.transformer import TransformerEncoderLayer


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = src == pad_idx
    tgt_padding_mask = tgt == pad_idx
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ConformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        old_kwargs = {k: v for k, v in kwargs.items() if "conv_" not in k}
        super().__init__(*args, **old_kwargs)
        del self.linear1
        del self.linear2
        self.conv1 = nn.Conv1d(
            kwargs["conv_in"],
            kwargs["conv_filter_size"],
            kernel_size=kwargs["conv_kernel"][0],
            padding=(kwargs["conv_kernel"][0] - 1) // 2,
        )
        self.conv2 = nn.Conv1d(
            kwargs["conv_filter_size"],
            kwargs["conv_in"],
            kernel_size=kwargs["conv_kernel"][1],
            padding=(kwargs["conv_kernel"][1] - 1) // 2,
        )

    def _ff_block(self, x):
        x = self.conv2(
            self.dropout(  # remove dropout for FastSpeech2
                self.activation(self.conv1(x.transpose(1, 2)))
            )
        ).transpose(1, 2)
        return self.dropout2(x)
