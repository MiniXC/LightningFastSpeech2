import os
import json

import pytorch_lightning as pl
from torch.nn.modules.transformer import TransformerEncoder
from model import ConformerEncoderLayer, PositionalEncoding, VarianceAdaptor
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import torch
import configparser
import multiprocessing
import wandb
from synthesiser import Synthesiser
from postnet import PostNet
import pandas as pd
from scipy.stats import ks_2samp

from dataset import ProcessedDataset, UnprocessedDataset

cpus = multiprocessing.cpu_count()

config = configparser.ConfigParser()
config.read("config.ini")


class FastSpeech2Loss(nn.Module):
    def __init__(self, postnet, use_snr=False):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.postnet = postnet
        self.use_snr = use_snr
        self.stochastic = config["model"].getboolean("duration_stochastic")

    @staticmethod
    def get_loss(pred, truth, loss, mask, unsqueeze=False):
        truth.requires_grad = False
        if unsqueeze:
            mask = mask.unsqueeze(-1)
        pred = pred.masked_select(mask)
        truth = truth.masked_select(mask)
        return loss(pred, truth)

    def forward(self, pred, truth, src_mask, tgt_mask, tgt_max_length=None):
        # TODO: do this via keys
        mel_pred = pred["mel"]
        pitch_pred = pred["pitch"]
        energy_pred = pred["energy"]
        duration_pred = pred["log_duration"]
        snr_pred = pred["snr"]
        postnet_pred = pred["postnet_output"]

        mel_tgt = truth["mel"]
        pitch_tgt = truth["pitch"]
        energy_tgt = truth["energy"]
        duration_tgt = truth["duration"]

        if self.use_snr:
            snr_tgt = truth["snr"]

        duration_tgt = torch.log(duration_tgt.float() + 1)

        src_mask = ~src_mask
        tgt_mask = ~tgt_mask

        if tgt_max_length is not None:
            mel_tgt = mel_tgt[:, :tgt_max_length, :]
            if config["dataset"].get("variance_level") == "frame":
                pitch_tgt = pitch_tgt[:, :tgt_max_length]
                energy_tgt = energy_tgt[:, :tgt_max_length]
                if self.use_snr:
                    snr_tgt = snr_tgt[:, :tgt_max_length]

        mel_loss = FastSpeech2Loss.get_loss(
            mel_pred, mel_tgt, self.l1_loss, tgt_mask, unsqueeze=True
        )

        if self.postnet:
            postnet_loss = FastSpeech2Loss.get_loss(
                mel_pred + postnet_pred, mel_tgt, self.l1_loss, tgt_mask, unsqueeze=True
            )

        if config["dataset"].get("variance_level") == "frame":
            pitch_energy_mask = tgt_mask
        elif config["dataset"].get("variance_level") == "phoneme":
            pitch_energy_mask = src_mask

        pitch_loss = FastSpeech2Loss.get_loss(
            pitch_pred, pitch_tgt, self.mse_loss, pitch_energy_mask
        )
        energy_loss = FastSpeech2Loss.get_loss(
            energy_pred, energy_tgt, self.mse_loss, pitch_energy_mask
        )
        if self.use_snr:
            snr_loss = FastSpeech2Loss.get_loss(
                snr_pred, snr_tgt, self.mse_loss, pitch_energy_mask
            )

        if not self.stochastic:
            duration_loss = FastSpeech2Loss.get_loss(
                duration_pred, duration_tgt, self.mse_loss, src_mask
            )
        else:
            duration_loss = torch.sum(duration_pred.float()) / 6000

        total_loss = mel_loss + pitch_loss + energy_loss + duration_loss

        if self.postnet:
            total_loss = total_loss + postnet_loss

        if self.use_snr:
            total_loss = total_loss + snr_loss

        result = [
            total_loss,
            mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        ]

        if self.postnet:
            result.append(postnet_loss)

        if self.use_snr:
            result.append(snr_loss)

        return result


class FastSpeech2(pl.LightningModule):
    def __init__(self, learning_rate=config["train"].getfloat("lr")):
        super().__init__()

        self.lr = learning_rate

        # data
        self.batch_size = config["train"].getint("batch_size")
        self.epochs = config["train"].getint("epochs")
        self.has_dvector = config["model"].getboolean("dvector")
        self.use_snr = config["model"].getboolean("snr")
        self.conditioned = config["model"].getboolean("conditioned")
        train_path = config["train"].get("train_path")
        valid_path = config["train"].get("valid_path")
        augment_duration = config["train"].getfloat("augment_duration")
        stats_path = config["train"].get("stats_path")
        with open(stats_path) as f:
            stats = json.load(f)

        if "+" in train_path:
            train_uds = [
                UnprocessedDataset(
                    x,
                    dvector=self.has_dvector,
                    augment_duration=augment_duration,
                    use_snr=self.use_snr,
                    conditioned=self.conditioned,
                )
                for x in train_path.split("+")
            ]
            train_dss = [
                ProcessedDataset(
                    unprocessed_ds=x, split="train", phone_vec=False, stats=stats
                )
                for x in train_uds
            ]
            self.train_ds = ConcatDataset(train_dss)
        else:
            train_ud = UnprocessedDataset(
                train_path,
                dvector=self.has_dvector,
                augment_duration=augment_duration,
                use_snr=self.use_snr,
                #max_entries=10_000,
                conditioned=self.conditioned,
            )
            self.train_ds = ProcessedDataset(
                unprocessed_ds=train_ud,
                split="train",
                phone_vec=False,
                stats=stats,
            )
        valid_ud = UnprocessedDataset(
            valid_path,
            dvector=self.has_dvector,
            augment_duration=0,
            use_snr=self.use_snr,
            #max_entries=500,
            conditioned=self.conditioned,
        )
        self.valid_ds = ProcessedDataset(
            unprocessed_ds=valid_ud,
            split="val",
            phone_vec=False,
            phone2id=self.train_ds.phone2id,
            stats=stats,
        )

        self.synth = Synthesiser(22050)

        vocab_n = self.train_ds.vocab_n
        speaker_n = self.train_ds.speaker_n
        stats = self.train_ds.stats

        # config
        encoder_hidden = config["model"].getint("encoder_hidden")
        encoder_head = config["model"].getint("encoder_head")
        encoder_layers = config["model"].getint("encoder_layers")
        encoder_dropout = config["model"].getfloat("encoder_dropout")
        decoder_hidden = config["model"].getint("decoder_hidden")
        decoder_head = config["model"].getint("decoder_head")
        decoder_layers = config["model"].getint("decoder_layers")
        decoder_dropout = config["model"].getfloat("decoder_dropout")
        conv_filter_size = config["model"].getint("conv_filter_size")
        kernel = (
            config["model"].getint("conv_kernel_1"),
            config["model"].getint("conv_kernel_2"),
        )
        self.tgt_max_length = config["model"].getint("tgt_max_length")
        self.max_lr = config["train"].getfloat("max_lr")
        mel_channels = config["model"].getint("mel_channels")
        self.has_postnet = config["model"].getboolean("postnet")

        # modules

        self.phone_embedding = nn.Embedding(vocab_n, encoder_hidden, padding_idx=0)

        if not self.has_dvector:
            self.speaker_embedding = nn.Embedding(
                speaker_n,
                encoder_hidden,
            )

        self.encoder = TransformerEncoder(
            ConformerEncoderLayer(
                encoder_hidden,
                encoder_head,
                conv_in=encoder_hidden,
                conv_filter_size=conv_filter_size,
                conv_kernel=kernel,
                batch_first=True,
                dropout=encoder_dropout,
            ),
            num_layers=encoder_layers,
        )
        self.positional_encoding = PositionalEncoding(
            encoder_hidden, dropout=encoder_dropout
        )
        self.variance_adaptor = VarianceAdaptor(stats, use_snr=self.use_snr)
        self.decoder = TransformerEncoder(
            ConformerEncoderLayer(
                decoder_hidden,
                decoder_head,
                conv_in=decoder_hidden,
                conv_filter_size=conv_filter_size,
                conv_kernel=kernel,
                batch_first=True,
                dropout=decoder_dropout,
            ),
            num_layers=decoder_layers,
        )

        self.linear = nn.Linear(
            decoder_hidden,
            mel_channels,
        )

        if self.has_postnet:
            self.postnet = PostNet()

        self.loss = FastSpeech2Loss(postnet=self.has_postnet, use_snr=self.use_snr)

        self.is_wandb_init = False

        if self.conditioned:
            nbins = 256
            self.pitch_bins = nn.Parameter(
                torch.linspace(stats["cond_pitch_min"], stats["cond_pitch_max"], nbins - 1),
                requires_grad=False,
            )
            self.condition_pitch_embedding = nn.Embedding(
                nbins,
                encoder_hidden,
            )
            self.energy_bins = nn.Parameter(
                torch.linspace(stats["cond_energy_min"], stats["cond_energy_max"], nbins - 1),
                requires_grad=False,
            )
            self.condition_energy_embedding = nn.Embedding(
                nbins,
                encoder_hidden,
            )
            self.duration_bins = nn.Parameter(
                torch.linspace(stats["cond_duration_min"], stats["cond_duration_max"], nbins - 1),
                requires_grad=False,
            )
            self.condition_duration_embedding = nn.Embedding(
                nbins,
                encoder_hidden,
            )
            if self.use_snr:
                self.snr_bins = nn.Parameter(
                    torch.linspace(-1.1, 1.1, nbins - 1),
                    requires_grad=False,
                )
                self.condition_snr_embedding = nn.Embedding(
                    nbins,
                    encoder_hidden,
                )

    def get_conditioned(self, targets):
        pitch_embedding = self.condition_pitch_embedding(torch.bucketize(targets["cond_pitch"], self.pitch_bins))
        energy_embedding = self.condition_energy_embedding(torch.bucketize(targets["cond_energy"], self.energy_bins))
        duration_embedding = self.condition_duration_embedding(torch.bucketize(targets["cond_duration"], self.duration_bins))
        if self.use_snr:
            snr_embedding = self.condition_snr_embedding(torch.bucketize(targets["cond_snr"], self.snr_bins))
            return pitch_embedding + energy_embedding + duration_embedding + snr_embedding
        else:
            return pitch_embedding + energy_embedding + duration_embedding

    def forward(self, targets, inference=False):
        phones = targets["phones"].to(self.device)
        speakers = targets["speaker"].to(self.device)

        def get_key(key):
            if not inference and key in targets and targets[key] is not None:
                return targets[key].to(self.device)
            else:
                return None

        pitch = get_key("pitch")
        energy = get_key("energy")
        duration = get_key("duration")
        snr = get_key("snr")

        src_mask = phones.eq(0)
        output = self.phone_embedding(phones)

        speakers = targets["speaker"]
        if not self.has_dvector:
            speaker_out = (
                self.speaker_embedding(speakers)
                .reshape(-1, 1, output.shape[-1])
                .repeat_interleave(phones.shape[1], dim=1)
            )
        else:
            speaker_out = (
                speakers.reshape(-1, 1, output.shape[-1])
                .repeat_interleave(phones.shape[1], dim=1)
                .to(self.device)
            )
        output = output + speaker_out

        # if self.conditioned:
        #     cond = self.get_conditioned(targets)
        #     cond = cond.reshape(-1, 1, output.shape[-1])
        #     cond = cond.repeat_interleave(phones.shape[1], dim=1)
        #     output = output + cond
        
        output = self.positional_encoding(output)
        output = self.encoder(output, src_key_padding_mask=src_mask)

        if not self.has_dvector:
            speaker_out2 = (
                self.speaker_embedding(speakers)
                .reshape(-1, 1, output.shape[-1])
                .repeat_interleave(phones.shape[1], dim=1)
            )
        else:
            speaker_out2 = (
                speakers.reshape(-1, 1, output.shape[-1])
                .repeat_interleave(phones.shape[1], dim=1)
                .to(self.device)
            )
        output = output + speaker_out2

        if self.conditioned:
            cond = self.get_conditioned(targets)
            cond = cond.reshape(-1, 1, output.shape[-1])
            cond = cond.repeat_interleave(phones.shape[1], dim=1)
            output = output + cond
        # speaker embedding addition was here

        if not self.has_dvector:
            variance_out = self.variance_adaptor(
                output, src_mask, pitch, energy, duration, snr, self.tgt_max_length, speaker_out2
            )
        else:
            variance_out = self.variance_adaptor(
                output, src_mask, pitch, energy, duration, snr, self.tgt_max_length, speaker_out2
            )
        output = variance_out["x"]
        output = self.positional_encoding(output)
        output = self.decoder(output, src_key_padding_mask=variance_out["tgt_mask"])

        output = self.linear(output)

        if self.has_postnet:
            postnet_output = self.postnet(output)
            final_output = postnet_output + output
        else:
            postnet_output = None
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
                    ks, pval = ks_2samp(var_df["real_val"].values, var_df["synth_val"].values)
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
            epochs=self.epochs,
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
            num_workers=cpus,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            collate_fn=self.valid_ds.collate_fn,
            num_workers=cpus,
        )
