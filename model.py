import math

import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.utils.rnn import pad_sequence
import configparser
from stochastic_duration_predictor.sdp import StochasticDurationPredictor

config = configparser.ConfigParser()
config.read("config.ini")


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


class VarianceAdaptor(nn.Module):
    def __init__(self, stats, use_snr=False):
        super().__init__()

        self.stochastic = False

        if config["model"].getboolean("duration_transformer"):
            self.duration_predictor = ConformerVariancePredictor()
        elif config["model"].getboolean("duration_stochastic"):
            self.duration_predictor = StochasticVariancePredictor()
            self.stochastic = True
        else:
            self.duration_predictor = VariancePredictor()
        self.length_regulator = LengthRegulator()
        self.pitch_encoder = VarianceEncoder()
        self.energy_encoder = VarianceEncoder()
        if use_snr:
            self.snr_encoder = VarianceEncoder(-1.1, 1.1)
        self.use_snr = use_snr

    def forward(
        self,
        x,
        src_mask,
        tgt_pitch=None,
        tgt_energy=None,
        tgt_duration=None,
        tgt_snr=None,
        tgt_max_length=None,
        speaker_embedding=None,
        c_pitch=1,
        c_energy=1,
        c_duration=1,
        c_snr=1,
    ):
        if not self.stochastic:
            duration_pred = self.duration_predictor(x, src_mask)
        else:
            if tgt_duration is not None:  # training
                duration_pred = self.duration_predictor(x, src_mask, tgt_duration, speaker_embedding)
            else:
                duration_pred = self.duration_predictor(x, src_mask, tgt_duration, speaker_embedding, inference=True)

        if config["dataset"].get("variance_level") == "phoneme":
            pitch_pred, pitch_out = self.pitch_encoder(x, tgt_pitch, src_mask, c_pitch)
            x = x + pitch_out
            energy_pred, energy_out = self.energy_encoder(
                x, tgt_energy, src_mask, c_energy
            )
            x = x + energy_out
            if self.use_snr:
                snr_pred, snr_out = self.snr_encoder(x, tgt_snr, src_mask, c_snr)
                x = x + snr_out

        if tgt_duration is not None:  # training
            duration_rounded = tgt_duration
        else:
            if not self.stochastic:
                duration_rounded = torch.round((torch.exp(duration_pred) - 1) * c_duration)
            else:
                duration_rounded = torch.ceil((torch.exp(duration_pred+1e-9)) * c_duration).masked_fill(duration_pred==0, 0)
            duration_rounded = torch.clamp(duration_rounded, min=0).int()

        x, tgt_len, tgt_mask = self.length_regulator(
            x, duration_rounded, tgt_max_length
        )

        if config["dataset"].get("variance_level") == "frame":
            pitch_pred, pitch_out = self.pitch_encoder(x, tgt_pitch, tgt_mask, c_pitch)
            energy_pred, energy_out = self.energy_encoder(
                x, tgt_energy, tgt_mask, c_energy
            )
            if self.use_snr:
                snr_pred, snr_out = self.snr_encoder(x, tgt_snr, tgt_mask, c_snr)
            if pitch_out.shape[1] > tgt_max_length:
                print("sample too long...")
            pitch_out = pitch_out[:, :tgt_max_length]
            energy_out = energy_out[:, :tgt_max_length]
            pitch_pred = pitch_pred[:, :tgt_max_length]
            energy_pred = energy_pred[:, :tgt_max_length]
            if self.use_snr:
                snr_out = snr_out[:, :tgt_max_length]
                snr_pred = snr_pred[:, :tgt_max_length]
            x = x + pitch_out + energy_out
            if self.use_snr:
                x = x + snr_out
        elif config["dataset"].get("variance_level") != "phoneme":
            raise ValueError("variance_level has to be frame or phoneme")

        # TODO: investigate if one should be applied before the other
        # ming implementation has pitch predicted first

        result = {
            "x": x,
            "pitch": pitch_pred,
            "energy": energy_pred,
            "log_duration": duration_pred,
            "duration_rounded": duration_rounded,
            "tgt_mask": tgt_mask,
        }

        if self.use_snr:
            result["snr"] = snr_pred

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
        return out, lengths, mask


class VarianceEncoder(nn.Module):
    def __init__(self, min=-1, max=1):
        super().__init__()

        n_bins = config["model"].getint("variance_nbins")
        encoder_hidden = config["model"].getint("encoder_hidden")

        self.predictor = VariancePredictor()
        self.bins = nn.Parameter(
            torch.linspace(min, max, n_bins - 1),
            requires_grad=False,
        )
        self.embedding = nn.Embedding(n_bins, encoder_hidden)

    def forward(self, x, tgt, mask, control):
        prediction = self.predictor(x, mask)
        if tgt is not None:
            embedding = self.embedding(torch.bucketize(tgt, self.bins))
        else:
            prediction = prediction * control
            embedding = self.embedding(torch.bucketize(prediction, self.bins))
        return prediction, embedding


class ConformerVariancePredictor(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()

        input_size = config["model"].getint("encoder_hidden")
        encoder_head = config["model"].getint("duration_transformer_head")
        dropout = config["model"].getfloat("variance_dropout")
        conv_filter_size = config["model"].getint("variance_filter_size")
        kernel = (
            config["model"].getint("conv_kernel_1"),
            config["model"].getint("conv_kernel_2"),
        )

        self.transformer = TransformerEncoder(
            ConformerEncoderLayer(
                input_size,
                encoder_head,
                conv_in=input_size,
                conv_filter_size=conv_filter_size,
                conv_kernel=kernel,
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=2,
        )
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, mask=None):
        out = self.linear(self.transformer(x))
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0)
        return out

class StochasticVariancePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = config["model"].getint("encoder_hidden")
        hidden_size = config["model"].getint("duration_stochastic_hidden")
        kernel = config["model"].getint("duration_stochastic_kernel")
        dropout = config["model"].getfloat("duration_stochastic_dropout")
        num_flows = config["model"].getint("duration_stochastic_num_flows")
        conditioning_size = config["model"].getint("duration_stochastic_conditioning_size")
        self.sigma = config["inference"].getfloat("duration_stochastic_sigma")

        self.sdp = StochasticDurationPredictor(
            input_size,
            hidden_size,
            kernel,
            dropout,
            num_flows,
            conditioning_size,
        )

    def forward(self, x, mask, tgt=None, condition=None, sigma=1.0, inference=False):
        if inference:
            sigma = self.sigma
        out = self.sdp(x, mask, tgt, g=condition, reverse=inference, noise_scale=sigma)
        if mask is not None and inference:
            out = out.masked_fill(mask, 0)
            #print(out)
        return out

class VariancePredictor(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()

        input_size = config["model"].getint("encoder_hidden")
        filter_size = config["model"].getint("variance_filter_size")
        kernel = config["model"].getint("variance_kernel")
        dropout = config["model"].getfloat("variance_dropout")
        depthwise = config["model"].getboolean("depthwise_conv")

        if not depthwise:
            self.layers = nn.Sequential(
                Transpose(
                    nn.Conv1d(input_size, filter_size, kernel, padding=(kernel - 1) // 2)
                ),
                activation(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
                Transpose(
                    nn.Conv1d(input_size, filter_size, kernel, padding=(kernel - 1) // 2)
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
                        nn.Conv1d(input_size, input_size, kernel, padding=(kernel - 1) // 2),
                        nn.Conv1d(input_size, filter_size, 1)
                    )
                ),
                activation(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout),
                Transpose(
                    nn.Sequential(
                        nn.Conv1d(input_size, input_size, kernel, padding=(kernel - 1) // 2),
                        nn.Conv1d(input_size, filter_size, 1)
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
