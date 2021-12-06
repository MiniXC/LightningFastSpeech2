import pytorch_lightning as pl
from torch.nn.modules.transformer import TransformerDecoder, TransformerEncoder
from model import ConformerEncoderLayer, PositionalEncoding, VarianceAdaptor
from torch import nn
import torch
import configparser

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
        mel_pred, pitch_pred, energy_pred, duration_pred = pred
        mel_tgt, pitch_tgt, energy_tgt, duration_tgt = truth
        duration_tgt = torch.log(duration_tgt.float() + 1)
        src_mask = ~src_mask
        tgt_mask = ~tgt_mask

        if tgt_max_length is not None:
            mel_tgt = mel_tgt[:, :tgt_max_length, :]

        mel_loss = FastSpeech2Loss.get_loss(mel_pred, mel_tgt, self.l1_loss, tgt_mask, unsqueeze=True)
        pitch_loss = FastSpeech2Loss.get_loss(
            pitch_pred, pitch_tgt, self.mse_loss, src_mask
        )
        energy_loss = FastSpeech2Loss.get_loss(
            energy_pred, energy_tgt, self.mse_loss, src_mask
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
    def __init__(self, vocab_n, speaker_n, stats, batch_size, dataset_size):
        super().__init__()

        self.batch_size = batch_size
        self.dataset_size = dataset_size

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
        self.epochs = config["train"].getint("epochs")
        mel_channels = config["model"].getint("mel_channels")

        self.phone_embedding = nn.Embedding(vocab_n, encoder_hidden, padding_idx=0)
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
        self.linear = nn.Linear(
            decoder_hidden,
            mel_channels,
        )

        self.loss = FastSpeech2Loss()

    def forward(self, phones, speakers, pitch, energy, duration):
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
            output,
            variance_out["pitch"],
            variance_out["energy"],
            variance_out["log_duration"],
        ), src_mask, variance_out["tgt_mask"]

    def training_step(self, batch):
        logits, src_mask, tgt_mask = self(
            batch["phones"],
            batch["speaker"],
            batch["pitch"],
            batch["energy"],
            batch["duration"],
        )
        truth = (batch["mel"], batch["pitch"], batch["energy"], batch["duration"])
        loss = self.loss(logits, truth, src_mask, tgt_mask, self.tgt_max_length)
        return loss[0]

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), self.max_lr)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.max_lr,
            steps_per_epoch=self.dataset_size//self.batch_size,
            epochs=self.epochs
        )

        sched = {
            'scheduler': self.scheduler,
            'interval': 'step',
        }

        return [self.optimizer], [sched]

