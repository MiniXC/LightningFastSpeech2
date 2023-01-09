from cmath import inf
import math

import numpy as np
import torch
import torch.nn as nn

from torch.nn import TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from einops.layers.torch import Rearrange, Reduce

from litfass.third_party.stochastic_duration_predictor.sdp import StochasticDurationPredictor
from litfass.dataset.cwt import CWT
from litfass.fastspeech2.utils import bucketize, Timer

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
        x = x + self.pe[:, : x.size(1), :].to(x.dtype)
        return self.dropout(x)


class Transpose(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.transpose(1, 2)).transpose(1, 2)


class LinearDummy():
    def __init__(self):
        self.weight = None
        self.bias = None

class ConformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        old_kwargs = {k: v for k, v in kwargs.items() if "conv_" not in k}
        super().__init__(*args, **old_kwargs)
        del self.linear1
        del self.linear2
        self.linear1 = LinearDummy()
        self.linear2 = LinearDummy()
        if "conv_depthwise" in kwargs and kwargs["conv_depthwise"]:
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    kwargs["conv_in"],
                    kwargs["conv_in"],
                    kernel_size=kwargs["conv_kernel"][0],
                    padding="same",
                    groups=kwargs["conv_in"],
                ),
                nn.Conv1d(kwargs["conv_in"], kwargs["conv_filter_size"], 1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    kwargs["conv_filter_size"],
                    kwargs["conv_filter_size"],
                    kernel_size=kwargs["conv_kernel"][1],
                    padding="same",
                    groups=kwargs["conv_in"],
                ),
                nn.Conv1d(kwargs["conv_filter_size"], kwargs["conv_in"], 1),
            )
        else:
            self.conv1 = nn.Conv1d(
                kwargs["conv_in"],
                kwargs["conv_filter_size"],
                kernel_size=kwargs["conv_kernel"][0],
                padding="same",
            )
            self.conv2 = nn.Conv1d(
                kwargs["conv_filter_size"],
                kwargs["conv_in"],
                kernel_size=kwargs["conv_kernel"][1],
                padding="same",
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x):
        x = self.conv2(
            self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        ).transpose(1, 2)
        return self.dropout2(x)


class SpeakerEmbedding(nn.Module):
    def __init__(self, embedding_dim, speaker_type, nspeakers=None):
        super().__init__()
        self.speaker_type = speaker_type
        self.embedding_dim = embedding_dim  
        if "dvector" in speaker_type:
            self.projection = nn.Linear(256, embedding_dim)
            self.has_projection = True
        elif speaker_type == "id":
            self.speaker_embedding = nn.Embedding(nspeakers, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x, input_length, output_shape):
        if self.has_projection:
            out = self.projection(x)
        else:
            out = self.speaker_embedding(x)
        out = self.relu(out)
        return out.reshape(-1, 1, output_shape).expand(-1, input_length, -1)


class PriorEmbedding(nn.Module):
    def __init__(self, embedding_dim, nbins, stats):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bins = nn.Parameter(
            torch.linspace(stats["min"], stats["max"], nbins - 1),
            requires_grad=False,
        )
        self.embedding = nn.Embedding(
            nbins,
            embedding_dim,
        )
        self.relu = nn.ReLU()

    def forward(self, x, input_length):
        out = self.relu(self.embedding(bucketize(x, self.bins)))
        return out.reshape(-1, 1, self.embedding_dim).expand(-1, input_length, -1)


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        stats,
        variances,
        variance_levels,
        variance_transforms,
        variance_nlayers,
        variance_kernel_size,
        variance_dropout,
        variance_filter_size,
        variance_nbins,
        variance_depthwise_conv,
        duration_nlayers,
        duration_stochastic,
        duration_kernel_size,
        duration_dropout,
        duration_filter_size,
        duration_depthwise_conv,
        encoder_hidden,
        max_length,
    ):
        super().__init__()
        self.variances = variances
        self.variance_levels = variance_levels
        self.variance_transforms = variance_transforms
        self.duration_stochastic = duration_stochastic
        self.max_length = max_length

        if self.duration_stochastic:
            if duration_depthwise_conv:
                raise NotImplementedError(
                    "Depthwise convolution not implemented for Flow-Based duration prediction"
                )
            self.duration_predictor = StochasticDurationPredictorWrapper(
                duration_nlayers,
                encoder_hidden,
                duration_filter_size,
                duration_kernel_size,
                duration_dropout,
            )
        else:
            self.duration_predictor = VariancePredictor(
                duration_nlayers,
                encoder_hidden,
                duration_filter_size,
                duration_kernel_size,
                duration_dropout,
                duration_depthwise_conv,
            )

        self.length_regulator = LengthRegulator(pad_to_multiple_of=256)

        self.encoders = {}
        for var in self.variances:
            self.encoders[var] = VarianceEncoder(
                variance_nlayers[variances.index(var)],
                encoder_hidden,
                variance_filter_size,
                variance_kernel_size[variances.index(var)],
                variance_dropout[variances.index(var)],
                variance_depthwise_conv,
                stats[var]["min"],
                stats[var]["max"],
                stats[var]["mean"],
                stats[var]["std"],
                variance_nbins,
                cwt=variance_transforms[variances.index(var)] == "cwt",
            )
        self.encoders = nn.ModuleDict(self.encoders)

        self.frozen_components = []

    def freeze(self, component):
        if component == "duration":
            for param in self.duration_predictor.parameters():
                param.requires_grad = False
        else:
            for param in self.encoders[component].parameters():
                param.requires_grad = False
        self.frozen_components.append(component)

    def forward(
        self,
        x,
        src_mask,
        targets,
        inference=False,
        tf_ratio=1.0,
        oracles=[],
    ):
        if not self.duration_stochastic:
            duration_pred = self.duration_predictor(x, src_mask)
        else:
            if not inference:
                duration_pred = self.duration_predictor(
                    x, src_mask, targets["duration"].to(x.device)
                )
            else:
                duration_pred = self.duration_predictor(
                    x, src_mask, inference=True
                )

        result = {}

        tf_val = np.random.uniform(0, 1) <= tf_ratio

        out_val = None

        for i, var in enumerate(self.variances):
            if self.variance_levels[i] == "phone":
                if (not inference and tf_val) or var in oracles:
                    if self.variance_transforms[i] == "cwt":
                        pred, out = self.encoders[var](
                            x, targets[f"variances_{var}_signal"], src_mask
                        )
                    else:
                        pred, out = self.encoders[var](
                            x, targets[f"variances_{var}"], src_mask
                        )
                else:
                    pred, out = self.encoders[var](x, None, src_mask)
                result[f"variances_{var}"] = pred
                if out_val is None:
                    out_val = out
                else:
                    out_val = out_val + out
                x = x + out

        if not inference:
            duration_rounded = targets["duration"]
        else:
            if not self.duration_stochastic:
                duration_rounded = torch.round((torch.exp(duration_pred) - 1))
            else:
                duration_rounded = torch.ceil(
                    (torch.exp(duration_pred + 1e-9))
                )
                duration_rounded[duration_pred == 0] = 0
            duration_rounded = torch.clamp(duration_rounded, min=0).int()
            duration_rounded = duration_rounded * (~src_mask).int()
            for i in range(len(duration_rounded)):
                if duration_rounded[i].sum() <= (~src_mask[i]).sum() // 2:
                    duration_rounded[i] = (~src_mask[i]).int()
                    print("Zero duration, setting to 1")
            duration_rounded = duration_rounded.cpu()

        if not inference:
            max_len = targets["mel"].shape[1]
        else:
            max_len = duration_rounded.sum(axis=1).max()

        x, tgt_mask = self.length_regulator(x, duration_rounded, max_len, targets["duration_mask"][0], targets["duration_mask"][1])#self.max_length)
        if out_val is not None:
            out_val, _ = self.length_regulator(out_val, duration_rounded, max_len, targets["duration_mask"][0], targets["duration_mask"][1])#self.max_length)

        for i, var in enumerate(self.variances):
            if self.variance_levels[i] == "frame":
                if (not inference and tf_val) or var in oracles:
                    if self.variance_transforms[i] == "cwt":
                        pred, out = self.encoders[var](
                            x, targets[f"variances_{var}_signal"], tgt_mask
                        )
                    else:
                        pred, out = self.encoders[var](
                            x, targets[f"variances_{var}"], tgt_mask
                        )
                else:
                    pred, out = self.encoders[var](x, None, tgt_mask)
                result[f"variances_{var}"] = pred
                if out_val is None:
                    out_val = out
                else:
                    out_val = out_val + out
                x = x + out

        result["x"] = x
        result["duration_prediction"] = duration_pred
        result["duration_rounded"] = duration_rounded
        result["tgt_mask"] = tgt_mask
        result["out"] = out_val

        return result


class LengthRegulator(nn.Module):
    def __init__(self, pad_to_multiple_of=None):
        super().__init__()
        self.pad_to_multiple_of = pad_to_multiple_of

    @staticmethod
    def repeat_batched(x, durations, target_length, bat_ind, val_ind, padding_value=0):
        durations = nn.ConstantPad1d((0, 1), 0)(durations)
        durations[:, -1] += target_length - durations.sum(axis=1)
        x = nn.ConstantPad1d((0, 0, 0, 1), padding_value)(x)
        
        # bat_ind = torch.arange(0, x.shape[0]).unsqueeze(-1).expand(-1, int(target_length)).flatten()
        # val_ind = torch.arange(0, x.shape[1]).repeat(x.shape[0])
        # flat_dur = durations.flatten()
        # val_ind = val_ind.flatten().repeat_interleave(durations.flatten(), dim=0)
        #print(~(val_ind.view(x.shape[0], -1)))

        tgt_mask = ~(val_ind.view(x.shape[0], -1) == durations.shape[1]-1)
        x = x[bat_ind, val_ind].view(x.shape[0], -1, x.shape[-1])
        
        return x, ~tgt_mask          

    def forward(self, x, durations, target_length=None, bat_ind=None, val_ind=None, max_length=None):
        with Timer("repeat_batched") as t:
            out, mask = LengthRegulator.repeat_batched(x, durations, target_length, bat_ind, val_ind)
            return out, mask.to(x.device)


class VarianceEncoder(nn.Module):
    def __init__(
        self,
        nlayers,
        in_channels,
        filter_size,
        kernel_size,
        dropout,
        depthwise,
        min,
        max,
        mean,
        std,
        nbins,
        cwt,
    ):
        super().__init__()
        self.cwt = cwt
        self.predictor = VariancePredictor(
            nlayers, in_channels, filter_size, kernel_size, dropout, depthwise, cwt
        )
        if cwt:
            min = np.log(min)
            max = np.log(max)
        self.bins = nn.Parameter(
            torch.linspace(min, max, nbins - 1),
            requires_grad=False,
        )
        self.embedding = nn.Embedding(nbins, in_channels)
        if cwt:
            self.mean_std_linear = nn.Linear(filter_size, 2)
            self.cwt_obj = CWT()

        self.mean = mean
        self.std = std

    def forward(self, x, tgt, mask, control=1.0):
        if not self.cwt:
            prediction = self.predictor(x, mask)
        else:
            prediction, out_conv = self.predictor(x, mask, return_conv=True)
            mean_std = self.mean_std_linear(torch.mean(out_conv, axis=1))
            mean, std = mean_std[:, 0], mean_std[:, 1]

        if tgt is not None:
            if self.cwt:
                tgt = torch.log(tgt)
            else:
                tgt = tgt * self.std + self.mean
            embedding = self.embedding(bucketize(tgt, self.bins).to(x.device))
        else:
            if self.cwt:
                tmp_prediction = []
                for i in range(len(prediction)):
                    tmp_prediction.append(
                        self.cwt_obj.recompose(prediction[i].T, mean[i], std[i])
                    )
                spectrogram = prediction
                prediction = torch.stack(tmp_prediction)
                bucket_prediction = prediction
            else:
                bucket_prediction = prediction * self.std + self.mean
            prediction = prediction * control
            embedding = self.embedding(
                bucketize(bucket_prediction, self.bins).to(x.device)
            )

        if not self.cwt:
            return prediction, embedding
        else:
            if tgt is not None:
                return (
                    {
                        "spectrogram": prediction,
                        "mean": mean,
                        "std": std,
                    },
                    embedding,
                )
            else:
                return (
                    {
                        "reconstructed_signal": torch.exp(prediction),
                        "spectrogram": spectrogram,
                        "mean": mean,
                        "std": std,
                    },
                    embedding,
                )

class StochasticDurationPredictorWrapper(nn.Module):
    def __init__(self, nlayers, in_channels, filter_size, kernel_size, dropout):
        super().__init__()

        self.sdp = StochasticDurationPredictor(
            in_channels,
            filter_size,
            kernel_size,
            dropout,
            nlayers,
        )

    def forward(self, x, mask, tgt=None, sigma=1.0, inference=False):
        out = self.sdp(x, mask, tgt, reverse=inference, noise_scale=sigma)
        if mask is not None and inference:
            out[mask] = 0
        return out


class VariancePredictor(nn.Module):
    def __init__(
        self,
        nlayers,
        in_channels,
        filter_size,
        kernel_size,
        dropout,
        depthwise=False,
        cwt=False,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            *[
                VarianceConvolutionLayer(
                    in_channels, filter_size, kernel_size, dropout, depthwise
                )
                for _ in range(nlayers)
            ]
        )

        self.cwt = cwt
        if not self.cwt:
            self.linear = nn.Linear(filter_size, 1)
        else:
            self.linear = nn.Linear(filter_size, 10)

    def forward(self, x, mask=None, return_conv=False):
        out_conv = self.layers(x)
        out = self.linear(out_conv)
        if not self.cwt:
            out = out.squeeze(-1)
        # else:
        #     mask = torch.stack([mask] * 10, dim=-1)
        # if mask is not None:
        #     out[mask] = 0
        if mask is not None:
            out = out * ~mask
        if return_conv:
            return out, out_conv
        else:
            return out

class VarianceConvolutionLayer(nn.Module):
    def __init__(self, in_channels, filter_size, kernel_size, dropout, depthwise):
        super().__init__()
        if not depthwise:
            self.layers = nn.Sequential(
                Transpose(
                    nn.Conv1d(
                        in_channels,
                        filter_size,
                        kernel_size,
                        padding=(kernel_size - 1) // 2,
                    )
                ),
                nn.ReLU(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
            )
        else:
            self.layers = nn.Sequential(
                Transpose(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels,
                            in_channels,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                            groups=in_channels,
                        ),
                        nn.Conv1d(in_channels, filter_size, 1),
                    )
                ),
                nn.ReLU(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.layers(x)