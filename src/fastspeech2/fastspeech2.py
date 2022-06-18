import os
import json
import multiprocessing

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.transformer import TransformerEncoder
import wandb
import pandas as pd
from scipy.stats import ks_2samp

from dataset import TTSDataset
from model import ConformerEncoderLayer, PositionalEncoding, VarianceAdaptor, PriorEmbedding, SpeakerEmbedding
from third_party.hifigan import Synthesiser
from loss import FastSpeech2Loss

num_cpus = multiprocessing.cpu_count()

class FastSpeech2(pl.LightningModule):
    def __init__(
        self,
        lr=5e-03,
        batch_size=6,
        speaker_type="dvector",  # "none", "id", "dvector"
        min_length=0.5,
        max_length=32,
        augment_duration=0.1,  # 0.1,
        variances=["pitch", "energy", "snr"],
        variance_levels=["phone", "phone", "phone"],
        variance_transforms=["cwt", "cwt", "cwt"],  # "cwt", "log", "none"
        priors=["pitch", "energy", "snr", "duration"],
        n_mels=80,
        train_ds_params=None,
        valid_ds_params=None,
        stochastic_duration=False,
        encoder_hidden=256,
        encoder_head=2,
        encoder_layers=4,
        encoder_dropout=0.1,
        encoder_kernel_sizes=[9, 9, 9, 9],
        decoder_hidden=256,
        decoder_head=2,
        decoder_layers=4,  # TODO: test with 6 layers
        decoder_dropout=0.1,
        decoder_kernel_sizes=[9, 9, 9, 9],
        conv_filter_size=1024,
        variance_nbins=256,
    ):
        super().__init__()

        # hparams
        self.save_hyperparameters(ignore=["train_ds_params", "valid_ds_params"])

        # data
        if train_ds_params is not None:
            train_ds_params["speaker_type"] = speaker_type
            train_ds_params["min_length"] = min_length
            train_ds_params["max_length"] = max_length
            train_ds_params["augment_duration"] = augment_duration
            train_ds_params["variances"] = variances
            train_ds_params["variance_levels"] = variance_levels
            train_ds_params["variance_transforms"] = variance_transforms
            train_ds_params["priors"] = priors
            train_ds_params["n_mels"] = n_mels
            self.train_ds = TTSDataset(**train_ds_params)
        if valid_ds_params is not None:
            self.valid_ds = self.train_ds.create_validation_dataset(**valid_ds_params)

        self.synth = Synthesiser()

        # needed for inference without a dataset
        self.stats = self.train_ds.stats
        self.phone2id = self.train_ds.phone2id
        if self.train_ds.speaker_type == "dvector":
            self.speaker2dvector = self.train_ds.speaker2dvector
        if self.train_ds.speaker_type == "id":
            self.speaker2id = self.train_ds.speaker2id

        self.phone_embedding = nn.Embedding(
            len(self.phone2id), self.hparams.encoder_hidden, padding_idx=0
        )

        if not self.has_dvector:
            self.speaker_embedding = nn.Embedding(
                len(self.speaker2id),
                self.hparams.encoder_hidden,
            )

        # encoder
        self.encoder = TransformerEncoder(
            ConformerEncoderLayer(
                self.hparams.encoder_hidden, self.hparams.encoder_head
            ),
            num_layers=self.hparams.encoder_layers,
        )
        for i in range(self.hparams.encoder_layers):
            self.encoder.layers[i] = ConformerEncoderLayer(
                self.hparams.encoder_hidden,
                self.hparams.encoder_head,
                conv_in=self.hparams.encoder_hidden,
                conv_filter_size=self.hparams.conv_filter_size,
                conv_kernel=(self.hparams.encoder_kernel_sizes[i], 1),
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
            self.hparams.stochastic_duration,
            self.hparams.variance_nbins,
            self.hparams.encoder_hidden,
        )

        # decoder
        self.decoder = TransformerEncoder(
            ConformerEncoderLayer(
                self.hparams.decoder_hidden,
                self.hparams.decoder_head,
            ),
            num_layers=self.hparams.decoder_layers,
        )
        for i in range(self.hparams.decoder_layers):
            self.decoder.layers[i] = ConformerEncoderLayer(
                self.hparams.decoder_hidden,
                self.hparams.decoder_head,
                conv_in=self.hparams.decoder_hidden,
                conv_filter_size=self.hparams.conv_filter_size,
                conv_kernel=(self.hparams.decoder_kernel_sizes[i], 1),
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
            )

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
                len(self.speaker2id)
            )

        # loss
        self.loss = FastSpeech2Loss(
            self.hparams.variances,
            self.hparams.variance_levels,
            self.hparams.variance_transforms
        )

    def on_load_checkpoint(self, checkpoint):
        self.stats = checkpoint["stats"]
        self.phone2id = checkpoint["phone2id"]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["stats"] = self.stats
        checkpoint["phone2id"] = self.phone2id

    def forward(self, targets, inference=False):
        phones = targets["phones"].to(self.device)
        speakers = targets["speaker"].to(self.device)

        src_mask = phones.eq(0)
        output = self.phone_embedding(phones)

        output = self.positional_encoding(output)
        output = self.encoder(output, src_key_padding_mask=src_mask)

        output += self.speaker_embedding(targets["speaker"], output.shape[1])

        for prior in self.hparams.priors:
            output += self.prior_embeddings[prior](targets[prior], output.shape[1])

        # if not self.has_dvector:
        #     variance_out = self.variance_adaptor(
        #         output,
        #         src_mask,
        #         pitch,
        #         energy,
        #         duration,
        #         snr,
        #         self.tgt_max_length,
        #         speaker_out2,
        #     )
        # else:
        #     variance_out = self.variance_adaptor(
        #         output,
        #         src_mask,
        #         pitch,
        #         energy,
        #         duration,
        #         snr,
        #         self.tgt_max_length,
        #         speaker_out2,
        #     )

        output = variance_out["x"]
        output = self.positional_encoding(output)
        output = self.decoder(output, src_key_padding_mask=variance_out["tgt_mask"])

        output = self.linear(output)

        final_output = output

        result = {
            "mel": output,
            "pitch": variance_out["pitch"],
            "energy": variance_out["energy"],
            "log_duration": variance_out["log_duration"],
            "duration_rounded": variance_out["duration_rounded"],
            "postnet_output": postnet_output,
            "final_output": final_output,
        }

        if self.use_snr:
            result["snr"] = variance_out["snr"]

        return (
            result,
            src_mask,
            variance_out["tgt_mask"],
        )

    def training_step(self, batch):
        logits, src_mask, tgt_mask = self(batch)
        loss = self.loss(logits, batch, src_mask, tgt_mask, self.tgt_max_length)
        log_dict = {
            "train/total_loss": loss[0].item(),
            "train/mel_loss": loss[1].item(),
            "train/pitch_loss": loss[2].item(),
            "train/energy_loss": loss[3].item(),
            "train/duration_loss": loss[4].item(),
        }
        if self.has_postnet:
            log_dict["train/postnet_loss"] = loss[5].item()
        if self.use_snr:
            log_dict["train/snr_loss"] = loss[-1].item()
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        return loss[0]

    def validation_step(self, batch, batch_idx):
        preds, src_mask, tgt_mask = self(batch)
        old_tgt_mask = tgt_mask
        loss = self.loss(preds, batch, src_mask, tgt_mask, self.tgt_max_length)
        log_dict = {
            "eval/total_loss": loss[0].item(),
            "eval/mel_loss": loss[1].item(),
            "eval/pitch_loss": loss[2].item(),
            "eval/energy_loss": loss[3].item(),
            "eval/duration_loss": loss[4].item(),
        }
        if self.has_postnet:
            log_dict["eval/postnet_loss"] = loss[5].item()
        if self.use_snr:
            log_dict["eval/snr_loss"] = loss[-1].item()
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )

        if batch_idx == 0:
            self.eval_log_data = []
            self.phone_dict = {
                "phone": [],
                "variance": [],  # can be duration, pitch, energy, snr
                "real_val": [],
                "synth_val": [],
            }

        preds, src_mask, tgt_mask = self(batch, inference=True)

        # get the real and synthesised values for each phone and add to phone_dict
        for i in range(len(batch["phones"])):
            phones = batch["phones"][i]
            for j, phone in enumerate(phones):
                variances = ["duration", "pitch", "energy"]
                if self.use_snr:
                    variances.append("snr")
                for var in variances:
                    self.phone_dict["phone"].append(
                        self.train_ds.id2phone[phone.item()]
                    )
                    self.phone_dict["variance"].append(var)
                    self.phone_dict["real_val"].append(batch[var][i][j].item())
                    if var == "duration":
                        var = "log_duration"
                        synth_val = torch.round(torch.exp(preds[var][i][j]) - 1).item()
                    else:
                        synth_val = preds[var][i][j].item()
                    self.phone_dict["synth_val"].append(synth_val)

        if (
            self.eval_log_data is not None
            and len(self.eval_log_data) < 10
            and self.trainer.is_global_zero
        ):
            if not self.is_wandb_init:
                wandb.init(project="LightningFastSpeech", group="DDP")
                self.is_wandb_init = True
            old_src_mask = src_mask
            # no teacher forcing here
            mels = preds["mel"].cpu()
            pitchs = preds["pitch"].cpu()
            energys = preds["energy"].cpu()
            durations = preds["duration_rounded"].cpu()
            final_mels = preds["final_output"].cpu()
            if self.use_snr:
                snrs = preds["snr"].cpu()
            for i in range(len(mels)):
                if len(self.eval_log_data) >= 10:
                    break
                mel = final_mels[i][~tgt_mask[i]]
                true_mel = batch["mel"][i][~old_tgt_mask[i]]
                if len(mel) == 0:
                    print(
                        "predicted 0 length output, this is normal at the beginning of training"
                    )
                    continue
                pred_dict = {
                    "mel": mel,
                    "pitch": pitchs[i],
                    "energy": energys[i],
                    "duration": durations[i][~src_mask[i]],
                    "phones": batch["phones"][i],
                }
                true_dict = {
                    "mel": true_mel.cpu(),
                    "pitch": batch["pitch"][i].cpu(),
                    "energy": batch["energy"][i].cpu(),
                    "duration": batch["duration"][i].cpu()[~old_src_mask[i]],
                    "phones": batch["phones"][i],
                }
                if self.use_snr:
                    pred_dict["snr"] = snrs[i]
                    true_dict["snr"] = batch["snr"].cpu()[i]
                pred_fig = self.valid_ds.plot(pred_dict)
                true_fig = self.valid_ds.plot(true_dict)
                pred_audio = self.synth(mel.to("cuda:0"))[0]
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

            df = pd.DataFrame(self.phone_dict)
            table_dict = {
                "phone": [],
                "variance": [],
                "mean_real": [],
                "mean_synth": [],
                "std_real": [],
                "std_synth": [],
                "MAE": [],
                "count": [],
                "ks_stat": [],
                "ks_pval": [],
            }
            for phone in df["phone"].unique():
                phone_df = df[df["phone"] == phone]
                for var in phone_df["variance"].unique():
                    var_df = phone_df[phone_df["variance"] == var]
                    table_dict["phone"].append(phone)
                    table_dict["variance"].append(var)
                    table_dict["mean_real"].append(var_df["real_val"].mean())
                    table_dict["mean_synth"].append(var_df["synth_val"].mean())
                    table_dict["std_real"].append(var_df["real_val"].std())
                    table_dict["std_synth"].append(var_df["synth_val"].std())
                    table_dict["MAE"].append(
                        (var_df["real_val"] - var_df["synth_val"]).abs().mean()
                    )
                    table_dict["count"].append(var_df.shape[0])
                    ks, pval = ks_2samp(
                        var_df["real_val"].values, var_df["synth_val"].values
                    )
                    table_dict["ks_stat"].append(ks)
                    table_dict["ks_pval"].append(pval)

            metric_df = pd.DataFrame(table_dict)
            table = wandb.Table(dataframe=metric_df)
            wandb.log({"phone metrics": table})

            variances = ["duration", "pitch", "energy"]
            if self.use_snr:
                variances.append("snr")

            for var in variances:
                var_df = metric_df[metric_df["variance"] == var]
                var_df = var_df[var_df["phone"] != "[SILENCE]"]
                wandb.log({f"eval/{var}_MAE": var_df["MAE"].mean()})
                wandb.log({f"eval/{var}_KS": var_df["ks_stat"].mean()})

    def configure_optimizers(self):
        # "betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01}
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            self.lr,
            # betas=[0.8, 0.99],
            # eps=1e-9,
            # weight_decay=0.01
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            steps_per_epoch=len(self.train_ds) // self.batch_size,
            epochs=self.trainer.max_epochs,
        )

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, self.epochs
        # )

        sched = {
            "scheduler": self.scheduler,
            "interval": "step",
        }

        return [self.optimizer], [sched]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=num_cpus,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            collate_fn=self.valid_ds.collate_fn,
            num_workers=num_cpus,
        )
