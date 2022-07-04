import os
import multiprocessing

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import wandb
import pandas as pd
import matplotlib.pyplot as plt

from dataset.datasets import TTSDataset
from .model import (
    ConformerEncoderLayer,
    PositionalEncoding,
    VarianceAdaptor,
    PriorEmbedding,
    SpeakerEmbedding,
)
from third_party.hifigan import Synthesiser
from .loss import FastSpeech2Loss
from .noam import NoamLR

num_cpus = multiprocessing.cpu_count()

class FastSpeech2(pl.LightningModule):
    def __init__(
        self,
        train_ds=None,
        valid_ds=None,
        lr=5e-05,
        warmup_steps=4000,
        batch_size=6,
        speaker_type="dvector",  # "none", "id", "dvector"
        min_length=0.5,
        max_length=32,
        augment_duration=0.1,  # 0.1,
        variances=["pitch", "energy", "snr"],
        variance_levels=["frame", "frame", "frame"],
        variance_transforms=["none", "none", "none"],  # "cwt", "log", "none"
        variance_nlayers=[5, 5, 5],
        variance_loss_weights=[1, 1, 1],
        variance_kernel_size=[3, 3, 3],
        variance_dropout=[0.5, 0.5, 0.5],
        variance_filter_size=256,
        variance_nbins=256,
        variance_depthwise_conv=True,
        duration_nlayers=2,
        duration_loss_weight=1,
        duration_stochastic=False,
        duration_kernel_size=3,
        duration_dropout=0.5,
        duration_filter_size=256,
        duration_depthwise_conv=True,
        priors=[],#["pitch", "energy", "snr", "duration"],
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
        encoder_kernel_sizes=[9, 9, 9, 9],
        encoder_dim_feedforward=None,
        encoder_conformer=True,
        encoder_depthwise_conv=True,
        decoder_hidden=256,
        decoder_head=2,
        decoder_layers=4,  # TODO: test with 6 layers
        decoder_dropout=0.1,
        decoder_kernel_sizes=[9, 9, 9, 9],
        decoder_dim_feedforward=None,
        decoder_conformer=True,
        decoder_depthwise_conv=True,
        conv_filter_size=1024,
        valid_nexamples=10,
        valid_example_directory=None,
    ):
        super().__init__()

        self.lr = lr
        self.warmup_steps = warmup_steps

        self.valid_nexamples = valid_nexamples
        self.valid_example_directory = valid_example_directory
        self.batch_size = batch_size

        # hparams
        self.save_hyperparameters(ignore=[
            "train_ds",
            "valid_ds",
            "train_ds_kwargs", 
            "valid_ds_kwargs", 
            "valid_nexamples", 
            "valid_example_directory",
            "batch_size",
        ])

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
            self.valid_ds = self.train_ds.create_validation_dataset(valid_ds, **valid_ds_kwargs)

        self.synth = Synthesiser()

        # needed for inference without a dataset
        if train_ds is not None:
            self.stats = self.train_ds.stats
            self.phone2id = self.train_ds.phone2id
            if self.train_ds.speaker_type == "dvector":
                self.speaker2dvector = self.train_ds.speaker2dvector
            if self.train_ds.speaker_type == "id":
                self.speaker2id = self.train_ds.speaker2id

        self.phone_embedding = nn.Embedding(
            len(self.phone2id), self.hparams.encoder_hidden, padding_idx=0
        )

        # encoder

        self.encoder = TransformerEncoder(
            ConformerEncoderLayer(
                self.hparams.encoder_hidden,
                self.hparams.encoder_head,
                conv_in=self.hparams.encoder_hidden,
                conv_filter_size=self.hparams.conv_filter_size,
                conv_kernel=(self.hparams.encoder_kernel_sizes[0], 1),
                batch_first=True,
                dropout=self.hparams.encoder_dropout,
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
                    conv_filter_size=self.hparams.conv_filter_size,
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
            self.hparams.max_length * self.hparams.sampling_rate / self.hparams.hop_length,
        ).to(self.device)

        # decoder

        self.decoder = TransformerEncoder(
            ConformerEncoderLayer(
                self.hparams.decoder_hidden,
                self.hparams.decoder_head,
                conv_in=self.hparams.decoder_hidden,
                conv_filter_size=self.hparams.conv_filter_size,
                conv_kernel=(self.hparams.decoder_kernel_sizes[0], 1),
                batch_first=True,
                dropout=self.hparams.decoder_dropout,
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
                    conv_filter_size=self.hparams.conv_filter_size,
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
            self.hparams.max_length * self.hparams.sampling_rate / self.hparams.hop_length,
            loss_weights,
        )

    def on_load_checkpoint(self, checkpoint):
        self.stats = checkpoint["stats"]
        self.phone2id = checkpoint["phone2id"]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["stats"] = self.stats
        checkpoint["phone2id"] = self.phone2id

    def forward(self, targets, inference=False):
        phones = targets["phones"].to(self.device)
        speakers = targets["speaker"]

        src_mask = phones.eq(0)

        output = self.phone_embedding(phones)

        output = self.positional_encoding(output)

        output = output + self.speaker_embedding(speakers, output.shape[1])

        output = self.encoder(output, src_key_padding_mask=src_mask)

        for prior in self.hparams.priors:
            output += self.prior_embeddings[prior](
                torch.tensor(targets[f"priors_{prior}"]).to(self.device), 
                output.shape[1]
            )

        variance_output = self.variance_adaptor(
            output,
            src_mask,
            targets,
            inference=inference,
        )

        output = variance_output["x"]

        output = self.positional_encoding(output)

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
        losses = self.loss(result, batch)
        log_dict = {
            f"train/{k}_loss": v.item()
            for k, v in losses.items()
        }
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        result = self(batch)
        losses = self.loss(result, batch)
        log_dict = {
            f"eval/{k}_loss": v.item()
            for k, v in losses.items()
        }
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )

        if batch_idx == 0:
            self.eval_log_data = []
            # self.variance_dict = {
            #     "duration": {"pred": [], "true": []}
            # }
            # for var in self.hparams.variances:
            #     self.variance_dict[var] = {"pred": [], "true": []}

        inference_result = self(batch, inference=True)

        # self.variance_dict["duration"]["pred"] += list(inference_result["duration_rounded"])
        # self.variance_dict["duration"]["true"] += list(batch["duration_rounded"])
        # for var in self.hparams.variances:
        #     self.variance_dict[var] += list(inference_result[var])

        if (
            self.eval_log_data is not None
            and len(self.eval_log_data) < self.valid_nexamples
            and self.trainer.is_global_zero
        ):
            for i in range(len(batch["mel"])):
                if len(self.eval_log_data) >= self.valid_nexamples:
                    break
                pred_mel = inference_result["mel"][i][~inference_result["tgt_mask"][i]].cpu()
                true_mel = batch["mel"][i][~result["tgt_mask"][i]].cpu()
                pred_dict = {
                    "mel": pred_mel,
                    "duration": inference_result["duration_rounded"][i][~inference_result["src_mask"][i]].cpu(),
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
                        pred_dict["variances"][var]["spectrogram"] = inference_result[f"variances_{var}"]["spectrogram"][i][~inference_result[mask][i]].cpu()
                        pred_dict["variances"][var]["original_signal"] = inference_result[f"variances_{var}"]["reconstructed_signal"][i][~inference_result[mask][i]].cpu()
                    else:
                        pred_dict["variances"][var] = inference_result[f"variances_{var}"][i][~inference_result[mask][i]].cpu()
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
                        true_dict["variances"][var]["spectrogram"] = batch[f"variances_{var}_spectrogram"][i][~result[mask][i]].cpu()
                        true_dict["variances"][var]["original_signal"] = batch[f"variances_{var}_original_signal"][i][~result[mask][i]].cpu()
                    else:
                        true_dict["variances"][var] = batch[f"variances_{var}"][i][~result[mask][i]].cpu()

                for prior in self.hparams.priors:
                    true_dict["priors"][prior] = batch[f"priors_{prior}"][i]
                    pred_dict["priors"][prior] = batch[f"priors_{prior}"][i]

                if pred_dict["duration"].sum() == 0:
                    print("WARNING: duration is zero (common at beginning of training)")
                else:
                    pred_fig = self.valid_ds.plot(pred_dict, show=False)
                    true_fig = self.valid_ds.plot(true_dict, show=False)
                    if self.valid_example_directory is not None:
                        pred_fig.save(
                            os.path.join(self.valid_example_directory, f"pred_{batch['id'][i]}.png")
                        )
                        true_fig.save(
                            os.path.join(self.valid_example_directory, f"true_{batch['id'][i]}.png")
                        )
                    pred_audio = self.synth(pred_mel.to("cuda:0"))[0]
                    true_audio = self.synth(true_mel.to("cuda:0"))[0]
                    self.eval_log_data.append(
                        [
                            batch["text"][i],
                            wandb.Image(pred_fig),
                            wandb.Image(true_fig),
                            wandb.Audio(pred_audio, sample_rate=22050),
                            wandb.Audio(true_audio, sample_rate=22050),
                        ]
                    )

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
            self.eval_log_data = None
            

    def configure_optimizers(self):
        # "betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01}
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.lr,
            # betas=[0.9, 0.98],
            # eps=1e-8,
            # weight_decay=0.01
        )

        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=self.hparams.lr,
        #     steps_per_epoch=len(self.train_ds) // self.batch_size,
        #     epochs=10,
        # )
        self.scheduler = NoamLR(self.optimizer, self.hparams.warmup_steps)

        sched = {
            "scheduler": self.scheduler,
            "interval": "step",
        }

        return [self.optimizer], [sched]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.train_ds._collate_fn,
            num_workers=num_cpus,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            collate_fn=self.valid_ds._collate_fn,
            num_workers=num_cpus,
        )
