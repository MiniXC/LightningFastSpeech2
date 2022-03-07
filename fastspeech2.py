import pytorch_lightning as pl
from torch.nn.modules.transformer import TransformerEncoder
from model import ConformerEncoderLayer, PositionalEncoding, VarianceAdaptor
from torch import nn
from torch.utils.data import DataLoader
import torch
import configparser
import multiprocessing
import wandb

from dataset import ProcessedDataset, UnprocessedDataset

cpus = multiprocessing.cpu_count()

config = configparser.ConfigParser()
config.read("config.ini")

# TODO:
# allow to replace with "real" conformer
# allow to replace with linear FFN with same number of params
# allow controls to be set to curves/arrays
# preprocess on frame level and allow phoneme level
# add option for postnet


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    @staticmethod
    def get_loss(pred, truth, loss, mask, unsqueeze=False):
        truth.requires_grad = False
        if unsqueeze:
            mask = mask.unsqueeze(-1)
        pred = pred.masked_select(mask)
        truth = truth.masked_select(mask)
        return loss(pred, truth)

    def forward(self, pred, truth, src_mask, tgt_mask, tgt_max_length=None):
        mel_pred, pitch_pred, energy_pred, duration_pred, _ = pred
        mel_tgt, pitch_tgt, energy_tgt, duration_tgt = truth
        duration_tgt = torch.log(duration_tgt.float() + 1)

        src_mask = ~src_mask
        tgt_mask = ~tgt_mask

        if tgt_max_length is not None:
            mel_tgt = mel_tgt[:, :tgt_max_length, :]
            #tgt_mask = tgt_mask[:, :tgt_max_length]
            if config["dataset"].get("variance_level") == "frame":
                pitch_tgt = pitch_tgt[:, :tgt_max_length]
                energy_tgt = energy_tgt[:, :tgt_max_length]

        mel_loss = FastSpeech2Loss.get_loss(
            mel_pred, mel_tgt, self.l1_loss, tgt_mask, unsqueeze=True
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

        duration_loss = FastSpeech2Loss.get_loss(
            duration_pred, duration_tgt, self.mse_loss, src_mask
        )

        total_loss = mel_loss + pitch_loss + energy_loss + duration_loss

        return (
            total_loss,
            mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )


class FastSpeech2(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()

        self.lr = learning_rate

        # data
        self.batch_size = config["train"].getint("batch_size")
        self.epochs = config["train"].getint("epochs")
        train_path = config["train"].get("train_path")
        valid_path = config["train"].get("valid_path")
        train_ud = UnprocessedDataset(train_path)
        valid_ud = UnprocessedDataset(valid_path)
        self.train_ds = ProcessedDataset(
            unprocessed_ds=train_ud,
            split="train",
            phone_vec=False
        )
        self.valid_ds = ProcessedDataset(
            unprocessed_ds=valid_ud,
            split="val",
            phone_vec=False,
            phone2id=self.train_ds.phone2id,
            stats=self.train_ds.stats
        )

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

        # modules
        vocab_n = 363
        self.phone_embedding = nn.Embedding(vocab_n, encoder_hidden, padding_idx=0)
        self.speaker_embedding = nn.Embedding(speaker_n, encoder_hidden,)
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
            encoder_layers,
        )
        self.positional_encoding = PositionalEncoding(
            encoder_hidden, dropout=encoder_dropout
        )
        self.variance_adaptor = VarianceAdaptor(stats)
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
            decoder_layers,
        )
        self.linear = nn.Linear(decoder_hidden, mel_channels,)

        self.loss = FastSpeech2Loss()

    def forward(self, phones, speakers, pitch=None, energy=None, duration=None):
        phones = phones.to(self.device)
        speakers = speakers.to(self.device)
        if pitch is not None:
            pitch = pitch.to(self.device)
        if energy is not None:
            energy = energy.to(self.device)
        if duration is not None:
            duration = duration.to(self.device)
        src_mask = phones.eq(0)
        output = self.phone_embedding(phones)
        output = self.positional_encoding(output)
        output = self.encoder(output, src_key_padding_mask=src_mask)
        speaker_out = (
            self.speaker_embedding(speakers)
            .reshape(-1, 1, output.shape[-1])
            .repeat_interleave(phones.shape[1], dim=1)
        )
        output = output + speaker_out
        variance_out = self.variance_adaptor(
            output, src_mask, pitch, energy, duration, self.tgt_max_length
        )
        output = variance_out["x"]
        output = self.positional_encoding(output)
        output = self.decoder(output, src_key_padding_mask=variance_out["tgt_mask"])
        output = self.linear(output)
        return (
            (
                output,
                variance_out["pitch"],
                variance_out["energy"],
                variance_out["log_duration"],
                variance_out["duration_rounded"],
            ),
            src_mask,
            variance_out["tgt_mask"],
        )

    def training_step(self, batch):
        logits, src_mask, tgt_mask = self(
            batch["phones"],
            batch["speaker"],
            batch["pitch"],
            batch["energy"],
            batch["duration"],
        )
        truth = (
            batch["mel"],
            batch["pitch"],
            batch["energy"],
            batch["duration"],
        )
        loss = self.loss(logits, truth, src_mask, tgt_mask, self.tgt_max_length)
        self.log_dict(
            {
                "train/total_loss": loss[0].item(),
                "train/mel_loss": loss[1].item(),
                "train/pitch_loss": loss[2].item(),
                "train/energy_loss": loss[3].item(),
                "train/duration_loss": loss[4].item(),
            },
            batch_size=self.batch_size,
        )
        return loss[0]

    def validation_step(self, batch, batch_idx):
        preds, src_mask, tgt_mask = self(
            batch["phones"],
            batch["speaker"],
            batch["pitch"],
            batch["energy"],
            batch["duration"],
        )
        truth = (
            batch["mel"],
            batch["pitch"],
            batch["energy"],
            batch["duration"],
        )
        loss = self.loss(preds, truth, src_mask, tgt_mask, self.tgt_max_length)
        self.log_dict(
            {
                "eval/total_loss": loss[0].item(),
                "eval/mel_loss": loss[1].item(),
                "eval/pitch_loss": loss[2].item(),
                "eval/energy_loss": loss[3].item(),
                "eval/duration_loss": loss[4].item(),
            },
            batch_size=self.batch_size,
        )
        if batch_idx == 0:
            old_src_mask = src_mask
            old_tgt_mask = tgt_mask
            preds, src_mask, tgt_mask = self(batch["phones"], batch["speaker"])
            mels, pitchs, energys, _, durations = [pred.cpu() for pred in preds]
            log_data = []
            for i in range(len(mels)):
                if i >= 10:
                    break
                mel = mels[i][~tgt_mask[i]]
                true_mel = batch["mel"][i][~old_tgt_mask[i]]
                if len(mel) == 0:
                    print('predicted 0 length output, this is normal at the beginning of training')
                    continue
                pred_fig = self.valid_ds.plot(
                    {
                        "mel": mel,
                        "pitch": pitchs[i],
                        "energy": energys[i],
                        "duration": durations[i][~src_mask[i]],
                        "phones": batch["phones"][i]
                    }
                )
                true_fig = self.valid_ds.plot(
                    {
                        "mel": true_mel.cpu(),
                        "pitch": batch["pitch"][i].cpu(),
                        "energy": batch["energy"][i].cpu(),
                        "duration": batch["duration"][i].cpu()[~old_src_mask[i]],
                        "phones": batch["phones"][i],
                    }
                )
                pred_audio = self.valid_ds.synthesise(mel.to("cuda:0"))[0]
                true_audio = self.valid_ds.synthesise(true_mel.to("cuda:0"))[0]
                log_data.append(
                    [
                        batch["text"][i],
                        wandb.Image(pred_fig),
                        wandb.Image(true_fig),
                        wandb.Audio(pred_audio, sample_rate=22050),
                        wandb.Audio(true_audio, sample_rate=22050),
                    ]
                )
            table = wandb.Table(
                data=log_data,
                columns=[
                    "text",
                    "predicted_mel",
                    "true_mel",
                    "predicted_audio",
                    "true_audio",
                ],
            )
            self.logger.experiment.log({"examples": table})
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), self.lr)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            steps_per_epoch=len(self.train_ds) // self.batch_size,
            epochs=self.epochs,
        )

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
