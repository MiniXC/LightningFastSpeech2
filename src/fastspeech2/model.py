import math

import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.utils.rnn import pad_sequence
from third_party.stochastic_duration_predictor.sdp import StochasticDurationPredictor


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


class Transpose(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.transpose(1, 2)).transpose(1, 2)


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

class SpeakerEmbedding(nn.Module):
    def __init__(self, embedding_dim, speaker_type, nspeakers=None):
        super().__init__()
        self.speaker_type = speaker_type
        self.embedding_dim = embedding_dim
        if speaker_type == "dvector":
            self.projection = nn.Linear(256, embedding_dim)
        elif speaker_type == "id":
            self.speaker_embedding(nspeakers)

    def forward(self, x, input_length):
        if self.speaker_type == "dvector":
            out = self.projection(x)
        elif self.speaker_type == "id":
            out = self.speaker_embedding(x)
        return (
            out
            .reshape(-1, 1, self.embedding_dim)
            .repeat_interleave(input_length, dim=1)
        )


class PriorEmbedding(nn.Module):
    def __init__(self, embedding_dim, nbins, stats):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bins = nn.Parameter(
            torch.linspace(
                stats["min"], stats["max"], nbins - 1
            ),
            requires_grad=False,
        )
        self.embedding = nn.Embedding(
            nbins,
            embedding_dim,
        )

    def forward(self, x, input_length):
        out = self.condition_pitch_embedding(
            torch.bucketize(x, self.bins)
        )
        return (
            out
            .reshape(-1, 1, self.embedding_dim)
            .repeat_interleave(input_length, dim=1)
        )

class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        stats,
        variances,
        variance_levels,
        variance_transforms,
        stochastic_duration,
        max_length,
        nbins,
        embedding_dim,
    ):
        super().__init__()
        self.variances = variances
        self.variance_levels = variance_levels
        self.variance_transforms = variance_transforms
        self.stochastic_duration = stochastic_duration
        self.max_length = max_length

        if self.stochastic_duration:
            self.duration_predictor = StochasticVariancePredictor()
        else:
            self.duration_predictor = VariancePredictor()

        self.length_regulator = LengthRegulator()
        
        self.encoders = {}
        for var in self.variances:
            self.encoders[var] = VarianceEncoder(
                stats[var]["min"],
                stats[var]["max"],
                nbins,
                embedding_dim,
            )

    def forward(
        self,
        x,
        src_mask,
        targets,
        teacher_forcing=True,
    ):
        if not self.stochastic:
            duration_pred = self.duration_predictor(x, src_mask)
        else:
            if teacher_forcing:
                duration_pred = self.duration_predictor(
                    x, src_mask, targets["duration"]
                )
            else:
                duration_pred = self.duration_predictor(
                    x, src_mask, inference=True
                )

        result = {}

        for i, var in enumerate(self.variances):
            if self.variance_levels[i] == "phone":
                if teacher_forcing:
                    pred, out = self.encoders[var](x, targets["pitch"], src_mask)
                else:
                    pred, out = self.encoders[var](x, None, src_mask)
                result[var] = pred
                x += out

        if teacher_forcing:  # training
            duration_rounded = targets["duration"]
        else:
            if not self.stochastic_duration:
                duration_rounded = torch.round(
                    (torch.exp(duration_pred) - 1)
                )
            else:
                duration_rounded = torch.ceil(
                    (torch.exp(duration_pred + 1e-9))
                ).masked_fill(duration_pred == 0, 0)
            duration_rounded = torch.clamp(duration_rounded, min=0).int()

        x, tgt_mask = self.length_regulator(
            x, duration_rounded, self.max_length
        )

        for i, var in enumerate(self.variances):
            if self.variance_levels[i] == "frame":
                if teacher_forcing:
                    pred, out = self.encoders[var](x, targets["pitch"], src_mask)
                else:
                    pred, out = self.encoders[var](x, None, src_mask)
                result[var] = pred
                x += out

        result["x"] = x
        result["duration_prediction"] = duration_pred
        result["duration_rounded"] = duration_rounded
        result["tgt_mask"] = tgt_mask

        return result


class LengthRegulator(nn.Module):
    def forward(self, x, durations, max_length=None):
        repeated_list = [
            torch.repeat_interleave(x[i], durations[i], dim=0)
            for i in range(x.shape[0])
        ]
        lengths = torch.tensor([t.shape[0] for t in repeated_list]).long()
        max_length = min(lengths.max(), max_length)
        mask = ~(
            torch.arange(max_length).expand(len(lengths), max_length)
            < lengths.unsqueeze(1)
        ).to(x.device)
        out = pad_sequence(repeated_list, batch_first=True, padding_value=0)
        if max_length is not None:
            out = out[:, :max_length]
        return out, mask


class VarianceEncoder(nn.Module):
    def __init__(self, min, max, nbins, encoder_hidden):
        super().__init__()

        self.predictor = VariancePredictor()
        self.bins = nn.Parameter(
            torch.linspace(min, max, nbins - 1),
            requires_grad=False,
        )
        self.embedding = nn.Embedding(nbins, encoder_hidden)

    def forward(self, x, tgt, mask, control=1.0):
        prediction = self.predictor(x, mask)
        if tgt is not None:
            embedding = self.embedding(torch.bucketize(tgt, self.bins))
        else:
            prediction = prediction * control
            embedding = self.embedding(torch.bucketize(prediction, self.bins))
        return prediction, embedding

class StochasticVariancePredictor(nn.Module):
    def __init__(self, encoder_hidden, hidden_dim, kernel, dropout, num_flows, conditioning_size):
        super().__init__()

        self.sdp = StochasticDurationPredictor(
            encoder_hidden,
            hidden_dim,
            kernel,
            dropout,
            num_flows,
            conditioning_size,
        )

    def forward(self, x, mask, tgt=None, condition=None, sigma=1.0, inference=False):
        out = self.sdp(x, mask, tgt, g=condition, reverse=inference, noise_scale=sigma)
        if mask is not None and inference:
            out = out.masked_fill(mask, 0)
        return out


class VariancePredictor(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()

        input_size = config["model"].getint("encoder_hidden")
        filter_size = config["model"].getint("variance_filter_size")
        kernel = config["model"].getint("variance_kernel")
        dropout = config["model"].getfloat("variance_dropout")
        depthwise = config["model"].getboolean("depthwise_conv")

        # add num_conv layers

        if not depthwise:
            self.layers = nn.Sequential(
                Transpose(
                    nn.Conv1d(
                        input_size, filter_size, kernel, padding=(kernel - 1) // 2
                    )
                ),
                activation(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
                Transpose(
                    nn.Conv1d(
                        input_size, filter_size, kernel, padding=(kernel - 1) // 2
                    )
                ),
                activation(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
                nn.Linear(filter_size, 1),
            )
        else:
            self.layers = nn.Sequential(
                Transpose(
                    nn.Sequential(
                        nn.Conv1d(
                            input_size, input_size, kernel, padding=(kernel - 1) // 2
                        ),
                        nn.Conv1d(input_size, filter_size, 1),
                    )
                ),
                activation(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
                Transpose(
                    nn.Sequential(
                        nn.Conv1d(
                            input_size, input_size, kernel, padding=(kernel - 1) // 2
                        ),
                        nn.Conv1d(input_size, filter_size, 1),
                    )
                ),
                activation(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
                nn.Linear(filter_size, 1),
            )

    def forward(self, x, mask=None):
        out = self.layers(x)
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0)
        return out
