import os
import multiprocessing
from pathlib import Path
import random
import string

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import KernelDensity

from dataset.datasets import TTSDataset
from .model import (
    ConformerEncoderLayer,
    PositionalEncoding,
    VarianceAdaptor,
    PriorEmbedding,
    SpeakerEmbedding,
)
from third_party.hifigan import Synthesiser
from third_party.softdtw import SoftDTW
from .loss import FastSpeech2Loss
from .noam import NoamLR

num_cpus = multiprocessing.cpu_count()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class FastSpeech2(pl.LightningModule):
    def __init__(
        self,
        train_ds=None,
        valid_ds=None,
        lr=1e-04,
        warmup_steps=4000,
        batch_size=6,
        speaker_type="dvector",  # "none", "id", "dvector"
        min_length=0.5,
        max_length=32,
        augment_duration=0.1,  # 0.1,
        variances=["pitch", "energy", "snr"],
        variance_levels=["frame", "frame", "frame"],
        variance_transforms=["cwt", "none", "none"],  # "cwt", "log", "none"
        variance_nlayers=[5, 5, 5],
        variance_loss_weights=[5e-2, 5e-2, 5e-2],
        variance_kernel_size=[3, 3, 3],
        variance_dropout=[0.5, 0.5, 0.5],
        variance_filter_size=256,
        variance_nbins=256,
        variance_depthwise_conv=True,
        duration_nlayers=2,
        duration_loss_weight=5e-1,
        duration_stochastic=False,
        duration_kernel_size=3,
        duration_dropout=0.5,
        duration_filter_size=256,
        duration_depthwise_conv=True,
        priors=[],  # ["pitch", "energy", "snr", "duration"],
        mel_loss_weight=1,
        n_mels=80,
        sampling_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        train_ds_kwargs=None,
        valid_ds_kwargs=None,
        encoder_hidden=256,
        encoder_head=2,
        encoder_layers=4,
        encoder_dropout=0.1,
        encoder_kernel_sizes=[5, 25, 13, 9],
        encoder_dim_feedforward=None,
        encoder_conformer=True,
        encoder_depthwise_conv=True,
        encoder_conv_filter_size=1024,
        decoder_hidden=256,
        decoder_head=2,
        decoder_layers=4,
        decoder_dropout=0.1,
        decoder_kernel_sizes=[17, 21, 9, 13],
        decoder_dim_feedforward=None,
        decoder_conformer=True,
        decoder_depthwise_conv=True,
        decoder_conv_filter_size=1024,
        valid_nexamples=10,
        valid_example_directory=None,
        variance_early_stopping="none",  # "mae", "js", "none"
        variance_early_stopping_patience=4,
        variance_early_stopping_directory="variance_encoders",
        tf_ratio=1.0,
        tf_linear_schedule=False,
        tf_linear_schedule_start=0,
        tf_linear_schedule_end=20,
        tf_linear_schedule_end_ratio=0.0,
        num_workers=num_cpus,
    ):
        super().__init__()

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_workers = num_workers

        self.valid_nexamples = valid_nexamples
        self.valid_example_directory = valid_example_directory
        self.batch_size = batch_size

        if variance_early_stopping != "none":
            letters = string.ascii_lowercase
            random_dir = "".join(random.choice(letters) for i in range(10))
            self.variance_encoder_dir = (
                Path(variance_early_stopping_directory) / random_dir
            )
            self.variance_encoder_dir.mkdir(parents=True, exist_ok=True)

        # hparams
        self.save_hyperparameters(
            ignore=[
                "train_ds",
                "valid_ds",
                "train_ds_kwargs",
                "valid_ds_kwargs",
                "valid_nexamples",
                "valid_example_directory",
                "batch_size",
                "variance_early_stopping_directory",
                "num_workers"
            ]
        )

        # data
        if train_ds is not None:
            if train_ds_kwargs is None:
                train_ds_kwargs = {}
            train_ds_kwargs["speaker_type"] = speaker_type
            train_ds_kwargs["min_length"] = min_length
            train_ds_kwargs["max_length"] = max_length
            train_ds_kwargs["augment_duration"] = augment_duration
            train_ds_kwargs["variances"] = variances
            train_ds_kwargs["variance_levels"] = variance_levels
            train_ds_kwargs["variance_transforms"] = variance_transforms
            train_ds_kwargs["priors"] = priors
            train_ds_kwargs["n_mels"] = n_mels
            train_ds_kwargs["sampling_rate"] = sampling_rate
            train_ds_kwargs["n_fft"] = n_fft
            train_ds_kwargs["win_length"] = win_length
            train_ds_kwargs["hop_length"] = hop_length
            self.train_ds = TTSDataset(train_ds, **train_ds_kwargs)
        if valid_ds is not None:
            if valid_ds_kwargs is None:
                valid_ds_kwargs = {}
            self.valid_ds = self.train_ds.create_validation_dataset(
                valid_ds, **valid_ds_kwargs
            )

        self.synth = Synthesiser(device=self.device)

        # needed for inference without a dataset
        if train_ds is not None:
            self.stats = self.train_ds.stats
            self.phone2id = self.train_ds.phone2id
            if self.train_ds.speaker_type == "dvector":
                self.speaker2dvector = self.train_ds.speaker2dvector
            if self.train_ds.speaker_type == "id":
                self.speaker2id = self.train_ds.speaker2id

        if hasattr(self, "phone2id"):
            self.phone_embedding = nn.Embedding(
                len(self.phone2id), self.hparams.encoder_hidden, padding_idx=0
            )

        # encoder

        self.encoder = TransformerEncoder(
            ConformerEncoderLayer(
                self.hparams.encoder_hidden,
                self.hparams.encoder_head,
                conv_in=self.hparams.encoder_hidden,
                conv_filter_size=self.hparams.encoder_conv_filter_size,
                conv_kernel=(self.hparams.encoder_kernel_sizes[0], 1),
                batch_first=True,
                dropout=self.hparams.encoder_dropout,
                conv_depthwise=self.hparams.encoder_depthwise_conv,
            ),
            num_layers=self.hparams.encoder_layers,
        )

        if self.hparams.encoder_conformer:
            if self.hparams.encoder_dim_feedforward is not None:
                print("encoder_dim_feedforward is ignored for conformer")
            for i in range(self.hparams.encoder_layers):
                self.encoder.layers[i] = ConformerEncoderLayer(
                    self.hparams.encoder_hidden,
                    self.hparams.encoder_head,
                    conv_in=self.hparams.encoder_hidden,
                    conv_filter_size=self.hparams.encoder_conv_filter_size,
                    conv_kernel=(self.hparams.encoder_kernel_sizes[i], 1),
                    batch_first=True,
                    dropout=self.hparams.encoder_dropout,
                    conv_depthwise=self.hparams.encoder_depthwise_conv,
                )
        else:
            for i in range(self.hparams.encoder_layers):
                self.encoder.layers[i] = TransformerEncoderLayer(
                    self.hparams.encoder_hidden,
                    self.hparams.encoder_head,
                    dim_feedforward=self.hparams.encoder_dim_feedforward,
                    batch_first=True,
                    dropout=self.hparams.encoder_dropout,
                )
        self.positional_encoding = PositionalEncoding(
            self.hparams.encoder_hidden, dropout=self.hparams.encoder_dropout
        )

        # variances
        if hasattr(self, "stats"):
            self.variance_adaptor = VarianceAdaptor(
                self.stats,
                self.hparams.variances,
                self.hparams.variance_levels,
                self.hparams.variance_transforms,
                self.hparams.variance_nlayers,
                self.hparams.variance_kernel_size,
                self.hparams.variance_dropout,
                self.hparams.variance_filter_size,
                self.hparams.variance_nbins,
                self.hparams.variance_depthwise_conv,
                self.hparams.duration_nlayers,
                self.hparams.duration_stochastic,
                self.hparams.duration_kernel_size,
                self.hparams.duration_dropout,
                self.hparams.duration_filter_size,
                self.hparams.duration_depthwise_conv,
                self.hparams.encoder_hidden,
                self.hparams.max_length
                * self.hparams.sampling_rate
                / self.hparams.hop_length,
            ).to(self.device)

        # decoder
        self.decoder = TransformerEncoder(
            ConformerEncoderLayer(
                self.hparams.decoder_hidden,
                self.hparams.decoder_head,
                conv_in=self.hparams.decoder_hidden,
                conv_filter_size=self.hparams.decoder_conv_filter_size,
                conv_kernel=(self.hparams.decoder_kernel_sizes[0], 1),
                batch_first=True,
                dropout=self.hparams.decoder_dropout,
                conv_depthwise=self.hparams.encoder_depthwise_conv,
            ),
            num_layers=self.hparams.decoder_layers,
        )
        if self.hparams.decoder_conformer:
            if self.hparams.decoder_dim_feedforward is not None:
                print("decoder_dim_feedforward is ignored for conformer")
            for i in range(self.hparams.decoder_layers):
                self.decoder.layers[i] = ConformerEncoderLayer(
                    self.hparams.decoder_hidden,
                    self.hparams.decoder_head,
                    conv_in=self.hparams.decoder_hidden,
                    conv_filter_size=self.hparams.decoder_conv_filter_size,
                    conv_kernel=(self.hparams.decoder_kernel_sizes[i], 1),
                    batch_first=True,
                    dropout=self.hparams.decoder_dropout,
                    conv_depthwise=self.hparams.decoder_depthwise_conv,
                )
        else:
            for i in range(self.hparams.decoder_layers):
                self.decoder.layers[i] = TransformerEncoderLayer(
                    self.hparams.decoder_hidden,
                    self.hparams.decoder_head,
                    dim_feedforward=self.hparams.decoder_dim_feedforward,
                    batch_first=True,
                    dropout=self.hparams.decoder_dropout,
                )

        self.linear = nn.Linear(
            self.hparams.decoder_hidden,
            self.hparams.n_mels,
        )

        # priors
        if hasattr(self, "stats"):
            self.prior_embeddings = {}
            for prior in self.hparams.priors:
                self.prior_embeddings[prior] = PriorEmbedding(
                    self.hparams.encoder_hidden,
                    self.hparams.variance_nbins,
                    self.stats[f"{prior}_prior"],
                ).to(self.device)
            self.prior_embeddings = nn.ModuleDict(self.prior_embeddings)

        # speaker
        if self.hparams.speaker_type == "dvector":
            self.speaker_embedding = SpeakerEmbedding(
                self.hparams.encoder_hidden,
                self.hparams.speaker_type,
            )
        elif self.hparams.speaker_type == "id":
            if hasattr(self, "speaker2id"):
                self.speaker_embedding = SpeakerEmbedding(
                    self.hparams.encoder_hidden,
                    self.hparams.speaker_type,
                    len(self.speaker2id),
                )

        # loss
        loss_weights = {
            "mel": self.hparams.mel_loss_weight,
            "duration": self.hparams.duration_loss_weight,
        }
        for i, var in enumerate(self.hparams.variances):
            loss_weights[var] = self.hparams.variance_loss_weights[i]
        self.loss = FastSpeech2Loss(
            self.hparams.variances,
            self.hparams.variance_levels,
            self.hparams.variance_transforms,
            self.hparams.duration_stochastic,
            self.hparams.max_length
            * self.hparams.sampling_rate
            / self.hparams.hop_length,
            loss_weights,
        )

    def on_load_checkpoint(self, checkpoint):
        self.stats = checkpoint["stats"]
        self.variance_adaptor = VarianceAdaptor(
            self.stats,
            self.hparams.variances,
            self.hparams.variance_levels,
            self.hparams.variance_transforms,
            self.hparams.variance_nlayers,
            self.hparams.variance_kernel_size,
            self.hparams.variance_dropout,
            self.hparams.variance_filter_size,
            self.hparams.variance_nbins,
            self.hparams.variance_depthwise_conv,
            self.hparams.duration_nlayers,
            self.hparams.duration_stochastic,
            self.hparams.duration_kernel_size,
            self.hparams.duration_dropout,
            self.hparams.duration_filter_size,
            self.hparams.duration_depthwise_conv,
            self.hparams.encoder_hidden,
            self.hparams.max_length
            * self.hparams.sampling_rate
            / self.hparams.hop_length,
        ).to(self.device)
        self.prior_embeddings = {}
        for prior in self.hparams.priors:
            self.prior_embeddings[prior] = PriorEmbedding(
                self.hparams.encoder_hidden,
                self.hparams.variance_nbins,
                self.stats[f"{prior}_prior"],
            ).to(self.device)
        self.prior_embeddings = nn.ModuleDict(self.prior_embeddings)
        self.phone2id = checkpoint["phone2id"]
        self.phone_embedding = nn.Embedding(
            len(self.phone2id), self.hparams.encoder_hidden, padding_idx=0
        )
        if "speaker2id" in checkpoint:
            self.speaker2id = checkpoint["speaker2id"]
            self.speaker_embedding = SpeakerEmbedding(
                self.hparams.encoder_hidden,
                self.hparams.speaker_type,
                len(self.speaker2id),
            )
        if "speaker2dvector" in checkpoint:
            self.speaker2dvector = checkpoint["speaker2dvector"]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["stats"] = self.stats
        checkpoint["phone2id"] = self.phone2id
        if hasattr(self, "speaker2id"):
            checkpoint["speaker2id"] = self.speaker2id
        if hasattr(self, "speaker2dvector"):
            checkpoint["speaker2dvector"] = self.speaker2dvector

    def forward(self, targets, inference=False):
        phones = targets["phones"].to(self.device)
        speakers = targets["speaker"]

        src_mask = phones.eq(0)

        output = self.phone_embedding(phones)

        output = self.positional_encoding(output)

        output = output + self.speaker_embedding(
            speakers, output.shape[1], output.shape[-1]
        )

        output = self.encoder(output, src_key_padding_mask=src_mask)

        for prior in self.hparams.priors:
            output += self.prior_embeddings[prior](
                torch.tensor(targets[f"priors_{prior}"]).to(self.device),
                output.shape[1],
            )

        tf_ratio = self.hparams.tf_ratio

        if self.hparams.tf_linear_schedule:
            if self.current_epoch > self.hparams.tf_linear_schedule_start:
                tf_ratio = tf_ratio - (
                    (tf_ratio - self.hparams.tf_linear_schedule_end_ratio)
                    * (self.current_epoch - self.hparams.tf_linear_schedule_start)
                    / (
                        self.hparams.tf_linear_schedule_end
                        - self.hparams.tf_linear_schedule_start
                    )
                )

        self.log("tf_ratio", tf_ratio)

        variance_output = self.variance_adaptor(
            output,
            src_mask,
            targets,
            inference=inference,
            tf_ratio=tf_ratio,
        )

        output = variance_output["x"]

        output = self.positional_encoding(output)

        for prior in self.hparams.priors:
            output += self.prior_embeddings[prior](
                torch.tensor(targets[f"priors_{prior}"]).to(self.device),
                output.shape[1],
            )

        output = output + self.speaker_embedding(
            speakers, output.shape[1], output.shape[-1]
        )

        output = self.decoder(output, src_key_padding_mask=variance_output["tgt_mask"])

        output = self.linear(output)

        result = {
            "mel": output,
            "duration_prediction": variance_output["duration_prediction"],
            "duration_rounded": variance_output["duration_rounded"],
            "src_mask": src_mask,
            "tgt_mask": variance_output["tgt_mask"],
        }

        for var in self.hparams.variances:
            result[f"variances_{var}"] = variance_output[f"variances_{var}"]

        return result

    def training_step(self, batch):
        result = self(batch)
        losses = self.loss(result, batch) #frozen_components=self.variance_adaptor.frozen_components)
        log_dict = {f"train/{k}_loss": v.item() for k, v in losses.items()}
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        result = self(batch)
        losses = self.loss(result, batch)
        log_dict = {f"eval/{k}_loss": v.item() for k, v in losses.items()}
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.eval_log_data = []
            self.results_dict = {
                "duration": {"pred": [], "true": []},
                "mel": {"pred": [], "true": []},
            }
            for var in self.hparams.variances:
                self.results_dict[var] = {"pred": [], "true": []}

        inference_result = self(batch, inference=True)

        if (
            self.eval_log_data is not None
            and len(self.eval_log_data) < self.valid_nexamples
            and self.trainer.is_global_zero
        ):
            left_to_add = self.valid_nexamples - len(self.eval_log_data)
            self._add_to_results_dict(inference_result, batch, result, left_to_add)
            self._log_table_to_wandb(inference_result, batch, result)

    def _log_table_to_wandb(self, inference_result, batch, result):
        for i in range(len(batch["mel"])):
            if len(self.eval_log_data) >= self.valid_nexamples:
                break
            pred_mel = inference_result["mel"][i][
                ~inference_result["tgt_mask"][i]
            ].cpu()
            true_mel = batch["mel"][i][~result["tgt_mask"][i]].cpu()
            pred_dict = {
                "mel": pred_mel,
                "duration": inference_result["duration_rounded"][i][
                    ~inference_result["src_mask"][i]
                ].cpu(),
                "phones": batch["phones"][i],
                "text": batch["text"][i],
                "variances": {},
                "priors": {},
            }
            for j, var in enumerate(self.hparams.variances):
                if self.hparams.variance_levels[j] == "phone":
                    mask = "src_mask"
                elif self.hparams.variance_levels[j] == "frame":
                    mask = "tgt_mask"
                if self.hparams.variance_transforms[j] == "cwt":
                    pred_dict["variances"][var] = {}
                    pred_dict["variances"][var]["spectrogram"] = inference_result[
                        f"variances_{var}"
                    ]["spectrogram"][i][~inference_result[mask][i]].cpu()
                    pred_dict["variances"][var]["original_signal"] = inference_result[
                        f"variances_{var}"
                    ]["reconstructed_signal"][i][~inference_result[mask][i]].cpu()
                else:
                    pred_dict["variances"][var] = inference_result[f"variances_{var}"][
                        i
                    ][~inference_result[mask][i]].cpu()
            true_dict = {
                "mel": true_mel,
                "duration": batch["duration"][i][~result["src_mask"][i]].cpu(),
                "phones": batch["phones"][i],
                "text": batch["text"][i],
                "variances": {},
                "priors": {},
            }
            for j, var in enumerate(self.hparams.variances):
                if self.hparams.variance_levels[j] == "phone":
                    mask = "src_mask"
                elif self.hparams.variance_levels[j] == "frame":
                    mask = "tgt_mask"
                if self.hparams.variance_transforms[j] == "cwt":
                    true_dict["variances"][var] = {}
                    true_dict["variances"][var]["spectrogram"] = batch[
                        f"variances_{var}_spectrogram"
                    ][i][~result[mask][i]].cpu()
                    true_dict["variances"][var]["original_signal"] = batch[
                        f"variances_{var}_original_signal"
                    ][i][~result[mask][i]].cpu()
                else:
                    true_dict["variances"][var] = batch[f"variances_{var}"][i][
                        ~result[mask][i]
                    ].cpu()

            for prior in self.hparams.priors:
                true_dict["priors"][prior] = batch[f"priors_{prior}"][i]
                pred_dict["priors"][prior] = batch[f"priors_{prior}"][i]

            if pred_dict["duration"].sum() == 0:
                print("WARNING: duration is zero (common at beginning of training)")
            else:
                pred_fig = self.valid_ds.plot(pred_dict, show=False)
                true_fig = self.valid_ds.plot(true_dict, show=False)
                if self.valid_example_directory is not None:
                    Path(self.valid_example_directory).mkdir(
                        parents=True, exist_ok=True
                    )
                    pred_fig.save(
                        os.path.join(
                            self.valid_example_directory, f"pred_{batch['id'][i]}.png"
                        )
                    )
                    true_fig.save(
                        os.path.join(
                            self.valid_example_directory, f"true_{batch['id'][i]}.png"
                        )
                    )
                pred_audio = self.synth(pred_mel.to(self.device).float())[0]
                true_audio = self.synth(true_mel.to(self.device).float())[0]
                self.eval_log_data.append(
                    [
                        batch["text"][i],
                        wandb.Image(pred_fig),
                        wandb.Image(true_fig),
                        wandb.Audio(pred_audio, sample_rate=22050),
                        wandb.Audio(true_audio, sample_rate=22050),
                    ]
                )

    def _add_to_results_dict(self, inference_result, batch, result, add_n):
        # duration
        self.results_dict["duration"]["pred"] += list(
            inference_result["duration_rounded"][~inference_result["src_mask"]]
        )[:add_n]
        self.results_dict["duration"]["true"] += list(
            batch["duration"][~result["src_mask"]]
        )[:add_n]

        # mel
        self.results_dict["mel"]["pred"] += list(
            inference_result["mel"][~inference_result["tgt_mask"]]
        )[:add_n]
        self.results_dict["mel"]["true"] += list(batch["mel"][~result["tgt_mask"]])[
            :add_n
        ]

        for i, var in enumerate(self.hparams.variances):
            if self.hparams.variance_levels[i] == "phone":
                mask = "src_mask"
            elif self.hparams.variance_levels[i] == "frame":
                mask = "tgt_mask"
            if self.hparams.variance_transforms[i] == "cwt":
                self.results_dict[var]["pred"] += list(
                    inference_result[f"variances_{var}"]["reconstructed_signal"][
                        ~inference_result[mask]
                    ]
                )[:add_n]
                self.results_dict[var]["true"] += list(
                    batch[f"variances_{var}_original_signal"][~result[mask]]
                )[:add_n]
            else:
                self.results_dict[var]["pred"] += list(
                    inference_result[f"variances_{var}"][~inference_result[mask]]
                )[:add_n]
                self.results_dict[var]["true"] += list(
                    batch[f"variances_{var}"][~result[mask]]
                )[:add_n]

    def validation_epoch_end(self, validation_step_outputs):
        if self.trainer.is_global_zero:
            wandb.init(project="FastSpeech2")
            table = wandb.Table(
                data=self.eval_log_data,
                columns=[
                    "text",
                    "predicted_mel",
                    "true_mel",
                    "predicted_audio",
                    "true_audio",
                ],
            )
            wandb.log({"examples": table})
            if (
                not hasattr(self, "best_variances")
                and self.hparams.variance_early_stopping
            ):
                self.best_variances = {}
            for key in self.results_dict.keys():
                self.results_dict[key]["pred"] = [
                    x.cpu().numpy() for x in self.results_dict[key]["pred"]
                ]
                self.results_dict[key]["true"] = [
                    x.cpu().numpy() for x in self.results_dict[key]["true"]
                ]
                if key != "mel":
                    pred_list = np.random.choice(
                        np.array(self.results_dict[key]["pred"]), size=500
                    ).reshape(-1, 1)
                    true_list = np.random.choice(
                        np.array(self.results_dict[key]["true"]), size=500
                    ).reshape(-1, 1)
                    kde_pred = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                        pred_list
                    )
                    kde_true = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                        true_list
                    )
                    min_val = min(min(pred_list), min(true_list))
                    max_val = max(max(pred_list), max(true_list))
                    arange = np.arange(
                        min_val, max_val, (max_val - min_val) / 100
                    ).reshape(-1, 1)
                    var_js = jensenshannon(
                        np.exp(kde_pred.score_samples(arange)),
                        np.exp(kde_true.score_samples(arange)),
                    )
                    var_mae = np.mean(
                        np.abs(
                            np.array(self.results_dict[key]["pred"])
                            - np.array(self.results_dict[key]["true"])
                        )
                    )
                    if (
                        self.hparams.variance_early_stopping != "none"
                        and not (
                            key in self.best_variances
                            and self.best_variances[key][1] == -1
                        )
                        and key != "duration" # TODO: add duration to early stopping
                    ):
                        if key not in self.best_variances:
                            if self.hparams.variance_early_stopping == "mae":
                                self.best_variances[key] = [var_mae, 1]
                            elif self.hparams.variance_early_stopping == "js":
                                self.best_variances[key] = [var_js, 1]
                            torch.save(
                                self.variance_adaptor.encoders[key].state_dict(),
                                self.variance_encoder_dir / f"{key}_encoder_best.pt",
                            )
                        else:
                            if (
                                var_js < self.best_variances[key][0]
                                and self.hparams.variance_early_stopping == "js"
                            ):
                                self.best_variances[key] = [var_js, 1]
                                torch.save(
                                    self.variance_adaptor.encoders[key].state_dict(),
                                    self.variance_encoder_dir
                                    / f"{key}_encoder_best.pt",
                                )
                            elif (
                                var_mae < self.best_variances[key][0]
                                and self.hparams.variance_early_stopping == "mae"
                            ):
                                self.best_variances[key] = [var_mae, 1]
                                torch.save(
                                    self.variance_adaptor.encoders[key].state_dict(),
                                    self.variance_encoder_dir
                                    / f"{key}_encoder_best.pt",
                                )
                            else:
                                self.best_variances[key][1] += 1
                            if (
                                self.hparams.variance_early_stopping_patience
                                <= self.best_variances[key][1]
                            ):
                                self.best_variances[key][1] = -1
                                self.variance_adaptor.encoders[key].load_state_dict(
                                    torch.load(
                                        self.variance_encoder_dir
                                        / f"{key}_encoder_best.pt"
                                    )
                                )
                                # freeze encoder
                                print(f"Freezing encoder {key}")
                                self.variance_adaptor.freeze(key)

                    self.log_dict({f"eval/jensenshannon_{key}": var_js})
                    self.log_dict({f"eval/mae_{key}": var_mae})
                else:
                    pred_res = np.concatenate(
                        [
                            np.array([x[i] for x in self.results_dict[key]["pred"]])
                            for i in range(self.hparams.n_mels)
                        ]
                    )
                    true_res = np.concatenate(
                        [
                            np.array([x[i] for x in self.results_dict[key]["true"]])
                            for i in range(self.hparams.n_mels)
                        ]
                    )
                    pred_list = np.random.choice(pred_res, size=500).reshape(-1, 1)
                    true_list = np.random.choice(true_res, size=500).reshape(-1, 1)
                    kde_pred = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                        pred_list
                    )
                    kde_true = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                        true_list
                    )
                    min_val = min(min(pred_list), min(true_list))
                    max_val = max(max(pred_list), max(true_list))
                    arange = np.arange(
                        min_val, max_val, (max_val - min_val) / 100
                    ).reshape(-1, 1)
                    mel_js = jensenshannon(
                        np.exp(kde_pred.score_samples(arange)),
                        np.exp(kde_true.score_samples(arange)),
                    )
                    mel_softdtw = SoftDTW(normalize=True)(
                        torch.tensor(self.results_dict[key]["pred"]).float(),
                        torch.tensor(self.results_dict[key]["true"]).float(),
                    )
                    self.log_dict(
                        {
                            f"eval/jensenshannon_{key}": mel_js,
                            f"eval/softdtw_{key}": mel_softdtw,
                        }
                    )
            self.eval_log_data = None

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.lr,
            betas=[0.9, 0.98],
            eps=1e-8,
            weight_decay=0.01,
        )

        self.scheduler = NoamLR(self.optimizer, self.hparams.warmup_steps)

        sched = {
            "scheduler": self.scheduler,
            "interval": "step",
        }

        return [self.optimizer], [sched]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FastSpeech2")
        parser.add_argument("--lr", type=float, default=2e-04)
        parser.add_argument("--warmup_steps", type=int, default=4000)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--speaker_type", type=str, default="dvector")
        parser.add_argument("--min_length", type=float, default=0.5)
        parser.add_argument("--max_length", type=float, default=32)
        parser.add_argument("--augment_duration", type=float, default=0.1)
        parser.add_argument("--variances", nargs="+", type=str, default=["pitch", "energy", "snr"])
        parser.add_argument("--variance_levels", nargs="+", type=str, default=["frame", "frame", "frame"])
        parser.add_argument("--variance_transforms", nargs="+", type=str, default=["cwt", "none", "none"])
        parser.add_argument("--variance_nlayers", nargs="+", type=int, default=[5, 5, 5])
        parser.add_argument("--variance_loss_weights", nargs="+", type=float, default=[1, 1e-1, 1e-1])
        parser.add_argument("--variance_kernel_size", nargs="+", type=int, default=[3, 3, 3])
        parser.add_argument("--variance_dropout", nargs="+", type=float, default=[0.5, 0.5, 0.5])
        parser.add_argument("--variance_filter_size", type=int, default=256)
        parser.add_argument("--variance_nbins", type=int, default=256)
        parser.add_argument("--variance_depthwise_conv", type=str2bool, default=True)
        parser.add_argument("--duration_nlayers", type=int, default=2)
        parser.add_argument("--duration_loss_weight", type=float, default=5e-1)
        parser.add_argument("--duration_stochastic", type=str2bool, default=False)
        parser.add_argument("--duration_kernel_size", type=int, default=3)
        parser.add_argument("--duration_dropout", type=float, default=0.5)
        parser.add_argument("--duration_filter_size", type=int, default=256)
        parser.add_argument("--duration_depthwise_conv", type=str2bool, default=True)
        parser.add_argument("--priors", nargs="+", type=str, default=[])
        parser.add_argument("--mel_loss_weight", type=float, default=1)
        parser.add_argument("--n_mels", type=int, default=80)
        parser.add_argument("--sampling_rate", type=int, default=22050)
        parser.add_argument("--n_fft", type=int, default=1024)
        parser.add_argument("--win_length", type=int, default=1024)
        parser.add_argument("--hop_length", type=int, default=256)
        parser.add_argument("--encoder_hidden", type=int, default=256)
        parser.add_argument("--encoder_head", type=int, default=2)
        parser.add_argument("--encoder_layers", type=int, default=4)
        parser.add_argument("--encoder_dropout", type=float, default=0.1)
        parser.add_argument("--encoder_kernel_sizes", nargs="+", type=int, default=[5, 25, 13, 9])
        parser.add_argument("--encoder_dim_feedforward", type=int, default=None)
        parser.add_argument("--encoder_conformer", type=str2bool, default=True)
        parser.add_argument("--encoder_depthwise_conv", type=str2bool, default=True)
        parser.add_argument("--encoder_conv_filter_size", type=int, default=1024)
        parser.add_argument("--decoder_hidden", type=int, default=256)
        parser.add_argument("--decoder_head", type=int, default=2)
        parser.add_argument("--decoder_layers", type=int, default=4)
        parser.add_argument("--decoder_dropout", type=float, default=0.1)
        parser.add_argument("--decoder_kernel_sizes", nargs="+", type=int, default=[17, 21, 9, 13])
        parser.add_argument("--decoder_dim_feedforward", type=int, default=None)
        parser.add_argument("--decoder_conformer", type=str2bool, default=True)
        parser.add_argument("--decoder_depthwise_conv", type=str2bool, default=True)
        parser.add_argument("--decoder_conv_filter_size", type=int, default=1024)
        parser.add_argument("--valid_nexamples", type=int, default=10)
        parser.add_argument("--valid_example_directory", type=str, default=None)
        parser.add_argument("--variance_early_stopping", type=str, default="none")
        parser.add_argument("--variance_early_stopping_patience", type=int, default=4)
        parser.add_argument("--variance_early_stopping_directory", type=str, default="variance_encoders")
        parser.add_argument("--tf_ratio", type=float, default=1.0)
        parser.add_argument("--tf_linear_schedule", type=str2bool, default=False)
        parser.add_argument("--tf_linear_schedule_start", type=int, default=0)
        parser.add_argument("--tf_linear_schedule_end", type=int, default=20)
        parser.add_argument("--tf_linear_schedule_end_ratio", type=float, default=0.0)
        parser.add_argument("--num_workers", type=int, default=num_cpus)
        return parent_parser

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parent_parser = TTSDataset.add_model_specific_args(parent_parser, "Train")
        parser = parent_parser.add_argument_group("Valid Dataset")
        parser.add_argument("--valid_max_entries", type=int, default=None)
        parser.add_argument("--valid_shuffle_seed", type=int, default=42)
        return parent_parser

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.train_ds._collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=5,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            collate_fn=self.valid_ds._collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=5,
        )
