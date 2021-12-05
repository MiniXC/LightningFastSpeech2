from pandas.core.indexing import convert_missing_indexer
import pytorch_lightning as pl
from torch.nn.modules.transformer import TransformerEncoder
from model import ConformerEncoderLayer, PositionalEncoding
from torch import nn
import configparser

config = configparser.ConfigParser()
config.read("config.ini")


class FastSpeech2(pl.LightningModule):
    def __init__(self, vocab_n, speaker_n):
        super().__init__()

        # config
        encoder_hidden = config["model"].getint("encoder_hidden")
        encoder_head = config["model"].getint("encoder_head")
        encoder_layers = config["model"].getint("encoder_layers")
        encoder_dropout = config["model"].getfloat("encoder_dropout")
        conv_filter_size = config["model"].getint("conv_filter_size")
        kernel = (
            config["model"].getint("conv_kernel_1"),
            config["model"].getint("conv_kernel_2"),
        )

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

    def forward(self, phones, speakers):
        src_mask = phones.eq(0)
        output = self.phone_embedding(phones)
        output = self.positional_encoding(output)
        output = self.encoder(output, src_key_padding_mask=src_mask)
        output = output + self.speaker_embedding(speakers).unsqueeze(1).expand(
            -1, phones.shape[1], -1
        )
        print(output)
        raise
