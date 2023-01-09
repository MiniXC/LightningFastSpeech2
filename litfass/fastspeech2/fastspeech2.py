import os
import multiprocessing
from pathlib import Path
import random
import string
import hashlib
import json
import pickle
from copy import copy

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm.auto import tqdm
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import KernelDensity
import wandb


from litfass.dataset.datasets import TTSDataset
from litfass.fastspeech2.model import (
    ConformerEncoderLayer,
    PositionalEncoding,
    VarianceAdaptor,
    PriorEmbedding,
    SpeakerEmbedding,
)
from litfass.fastspeech2.fastdiff_variances import FastDiffVarianceAdaptor, FastDiffSpeakerGenerator
from litfass.third_party.hifigan import Synthesiser
from litfass.third_party.softdtw import SoftDTW
from litfass.third_party.argutils import str2bool
from litfass.third_party.fastdiff import FastDiff
from litfass.fastspeech2.log_gmm import LogGMM
from litfass.fastspeech2.loss import FastSpeech2Loss
from litfass.fastspeech2.noam import NoamLR
from litfass.fastspeech2.utils import Timer

num_cpus = multiprocessing.cpu_count()


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
        layer_dropout=0.1,
        variances=["pitch", "energy", "snr"],
        variance_levels=["frame", "frame", "frame"],
        variance_transforms=["cwt", "none", "none"],  # "cwt", "log", "none"
        variance_losses=["mse", "mse", "mse"],
        variance_nlayers=[5, 5, 5, 5],
        variance_loss_weights=[5e-2, 5e-2, 5e-2, 5e-2],
        variance_kernel_size=[3, 3, 3, 3],
        variance_dropout=[0.5, 0.5, 0.5, 0.5],
        variance_filter_size=256,
        variance_nbins=256,
        variance_depthwise_conv=True,
        duration_nlayers=2,
        duration_loss="mse",
        duration_loss_weight=5e-1,
        duration_stochastic=False,
        duration_kernel_size=3,
        duration_dropout=0.5,
        duration_filter_size=256,
        duration_depthwise_conv=True,
        mel_loss="l1",
        soft_dtw_gamma=0.1,
        soft_dtw_chunk_size=256,
        speaker_embedding_every_layer=False,
        prior_embedding_every_layer=False,
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
        num_workers=num_cpus,
        cache_path=None,
        priors_gmm=False,
        priors_gmm_max_components=5,
        priors_gmm_min_samples_per_component=20,
        priors_gmm_reg_covar=1e-3,
        priors_gmm_logs=[0,1,2,3],
        dvector_gmm=False,
        fastdiff_model=None,
        fastdiff_schedule=[0,1],
        fastdiff_schedule_start=0,
        fastdiff_schedule_end=20,
        sort_data_by_length=False,
        fastdiff_variances=True,
        fastdiff_speakers=False,
        fastdiff_speakers_loss_weight=1,
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
                "num_workers",
                "fastdiff_model",
            ]
        )

        self.eval_log_data = None

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
            if cache_path is not None:
                if isinstance(train_ds, list):
                    hashes = [x.hash for x in train_ds]
                else:
                    hashes = [train_ds.hash]
                kwargs = copy(train_ds_kwargs)
                kwargs.update({"hashes": hashes})
                ds_hash = hashlib.md5(
                    json.dumps(kwargs, sort_keys=True).encode("utf-8")
                ).hexdigest()
                cache_path_full = Path(cache_path) / f"train-full-{ds_hash}.pt"
                if cache_path_full.exists():
                    with cache_path_full.open("rb") as f:
                        self.train_ds = pickle.load(f)
                else:
                    self.train_ds = TTSDataset(train_ds, **train_ds_kwargs)
                    self.train_ds.hash = ds_hash
                    with open(cache_path_full, "wb") as f:
                        pickle.dump(self.train_ds, f)
            else:
                self.train_ds = TTSDataset(train_ds, **train_ds_kwargs)
        if valid_ds is not None:
            if valid_ds_kwargs is None:
                valid_ds_kwargs = {}
            if cache_path is not None:
                kwargs = copy(valid_ds_kwargs)
                kwargs.update({"hashes": [self.train_ds.hash, valid_ds.hash]})
                ds_hash = hashlib.md5(
                    json.dumps(kwargs, sort_keys=True).encode("utf-8")
                ).hexdigest()
                cache_path_full = Path(cache_path) / f"valid-full-{ds_hash}.pt"
                if cache_path_full.exists():
                    with cache_path_full.open("rb") as f:
                        self.valid_ds = pickle.load(f)
                else:
                    self.valid_ds = self.train_ds.create_validation_dataset(
                        valid_ds, **valid_ds_kwargs
                    )
                    self.valid_ds.hash = ds_hash
                    with open(cache_path_full, "wb") as f:
                        pickle.dump(self.valid_ds, f)
            else:
                self.valid_ds = self.train_ds.create_validation_dataset(
                    valid_ds, **valid_ds_kwargs
                )

        if fastdiff_model is None:
            self.synth = Synthesiser(device=self.device)

        # needed for inference without a dataset
        if train_ds is not None:
            self.stats = self.train_ds.stats
            self.phone2id = self.train_ds.phone2id
            if "dvector" in self.train_ds.speaker_type:
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
            # layer_drop_prob=self.hparams.layer_dropout,
        )

        if self.hparams.fastdiff_speakers:
            self.fastdiff_speaker_generator = FastDiffSpeakerGenerator(
                512, # hidden dim
                self.hparams.encoder_hidden, # conditional dim
                self.hparams.encoder_hidden, # speaker dim
            )
        else:
            self.fastdiff_speaker_generator = None

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
            if self.hparams.fastdiff_variances:
                self.variance_adaptor = FastDiffVarianceAdaptor(
                    self.stats,
                    self.hparams.variances,
                    self.hparams.variance_nlayers,
                    self.hparams.variance_kernel_size,
                    self.hparams.variance_dropout,
                    self.hparams.variance_filter_size,
                    self.hparams.variance_nbins,
                    self.hparams.variance_depthwise_conv,
                    self.hparams.duration_nlayers,
                    self.hparams.duration_kernel_size,
                    self.hparams.duration_dropout,
                    self.hparams.duration_filter_size,
                    self.hparams.duration_depthwise_conv,
                    self.hparams.encoder_hidden,
                    self.hparams.max_length
                    * self.hparams.sampling_rate
                    / self.hparams.hop_length,
                )
            else:
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
            # layer_drop_prob=self.hparams.layer_dropout,
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

        if fastdiff_model is not None:
            self.fastdiff_model = fastdiff_model
            self.fastdiff_model.to(self.device)
            self.fastdiff_linear = nn.Sequential(
                nn.Linear(
                    self.hparams.decoder_hidden,
                    self.hparams.decoder_hidden,
                ),
                nn.Linear(
                    self.hparams.decoder_hidden,
                    self.hparams.n_mels,
                ),
            )
            self.fastdiff_schedule = np.zeros(self.hparams.fastdiff_schedule_end)
            # linear
            self.fastdiff_schedule[
                self.hparams.fastdiff_schedule_start:self.hparams.fastdiff_schedule_end+1
            ] = np.linspace(
                self.hparams.fastdiff_schedule[0], 
                self.hparams.fastdiff_schedule[1], 
                self.hparams.fastdiff_schedule_end-self.hparams.fastdiff_schedule_start
            ) 
        else:
            self.fastdiff_model = None

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
        if "dvector" in self.hparams.speaker_type:
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

        if hasattr(self, "train_ds") and self.train_ds is not None and hasattr(self.train_ds, "speaker_priors"):
            self.speaker2priors = self.train_ds.speaker_priors

        # loss
        loss_weights = {
            "mel": self.hparams.mel_loss_weight,
            "duration": self.hparams.duration_loss_weight,
            "speakers": self.hparams.fastdiff_speakers_loss_weight
        }
        for i, var in enumerate(self.hparams.variances):
            loss_weights[var] = self.hparams.variance_loss_weights[i]
        if self.fastdiff_model is not None:
            fastdiff_loss = "mse"
            loss_weights["fastdiff"] = 1.0
        else:
            fastdiff_loss = None
        self.loss = FastSpeech2Loss(
            self.hparams.variances,
            self.hparams.variance_levels,
            self.hparams.variance_transforms,
            self.hparams.variance_losses,
            self.hparams.mel_loss,
            self.hparams.duration_loss,
            self.hparams.duration_stochastic,
            self.hparams.max_length
            * self.hparams.sampling_rate
            / self.hparams.hop_length,
            loss_weights,
            self.hparams.soft_dtw_gamma,
            self.hparams.soft_dtw_chunk_size,
            fastdiff_loss,
            fastdiff_variances=self.hparams.fastdiff_variances,
        )

        if (
            len(self.hparams.priors) > 0 and
            self.hparams.priors_gmm and
            hasattr(self, "speaker2priors")
        ):
            self._fit_speaker_prior_gmms()
        
        if (
            self.hparams.dvector_gmm and
            hasattr(self, "train_ds") and
            self.train_ds is not None and
            hasattr(self.train_ds, "speaker2dvector")
        ):
            self._fit_speaker_dvector_gmms()

        elif self.hparams.dvector_gmm:
            print("dvector_gmm is ignored because train_ds is not provided or speaker2dvector is not provided")

    def _fit_speaker_dvector_gmms(self):
        self.dvector_gmms = {}
        for speaker, dvector in self.train_ds.get_speaker_dvectors():
            self.dvector_gmms[speaker] = LogGMM(
                n_components=10,
                random_state=0
            )
            self.dvector_gmms[speaker].fit(dvector)

    def _fit_speaker_prior_gmms(self):
        self.speaker_gmms = {}
        reg_covar = self.hparams.priors_gmm_reg_covar
        for speaker in tqdm(self.speaker2priors.keys(), desc="fitting speaker prior gmms"):
            priors = self.speaker2priors[speaker]
            X = np.stack([priors[prior] for prior in self.hparams.priors], axis=1)
            gmm_kwargs = {
                "n_components": 1,
                "random_state": 0,
                "reg_covar": reg_covar,
                "logs": self.hparams.priors_gmm_logs,
            }
            best_bic = np.infty
            while True:
                gmm = LogGMM(**gmm_kwargs)
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                else:
                    break
                n_components = gmm_kwargs["n_components"]
                if n_components == len(X) or n_components >= self.hparams.priors_gmm_max_components or self.hparams.priors_gmm_min_samples_per_component * n_components > len(X):
                    break
                gmm_kwargs["n_components"] += 1
            self.speaker_gmms[speaker] = best_gmm
        pickle.dump(self.speaker_gmms, open(f"prior_gmms.pkl", "wb"))

    def on_load_checkpoint(self, checkpoint):
        self.stats = checkpoint["stats"]
        if not hasattr(self, "variance_adaptor"):
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
        if not hasattr(self, "prior_embeddings"):
            self.prior_embeddings = {}
            for prior in self.hparams.priors:
                self.prior_embeddings[prior] = PriorEmbedding(
                    self.hparams.encoder_hidden,
                    self.hparams.variance_nbins,
                    self.stats[f"{prior}_prior"],
                ).to(self.device)
            self.prior_embeddings = nn.ModuleDict(self.prior_embeddings)
        self.phone2id = checkpoint["phone2id"]
        if not hasattr(self, "phone_embedding"):
            self.phone_embedding = nn.Embedding(
                len(self.phone2id), self.hparams.encoder_hidden, padding_idx=0
            )
        if "speaker2id" in checkpoint:
            self.speaker2id = checkpoint["speaker2id"]
            if not hasattr(self, "speaker_embedding"):
                self.speaker_embedding = SpeakerEmbedding(
                    self.hparams.encoder_hidden,
                    self.hparams.speaker_type,
                    len(self.speaker2id),
                )
        if "speaker2dvector" in checkpoint:
            self.speaker2dvector = checkpoint["speaker2dvector"]
        if "speaker2priors" in checkpoint:
            self.speaker2priors = checkpoint["speaker2priors"]
        if "speaker_gmms" in checkpoint:
            self.speaker_gmms = checkpoint["speaker_gmms"]
        if "dvector_gmms" in checkpoint:
            self.dvector_gmms = checkpoint["dvector_gmms"]

        if any(["fastdiff_model" in x for x in checkpoint["state_dict"].keys()]):
            self.fastdiff_model = FastDiff.FastDiff()
            self.fastdiff_linear = nn.Sequential(
                nn.Linear(
                    self.hparams.decoder_hidden,
                    self.hparams.decoder_hidden,
                ),
                nn.Linear(
                    self.hparams.decoder_hidden,
                    self.hparams.n_mels,
                ),
            )

        # drop shape mismatched layers
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            # pylint: disable=unsupported-membership-test,unsubscriptable-object
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            # pylint: enable=unsupported-membership-test,unsubscriptable-object
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["stats"] = self.stats
        checkpoint["phone2id"] = self.phone2id
        if hasattr(self, "speaker2id"):
            checkpoint["speaker2id"] = self.speaker2id
        if hasattr(self, "speaker2dvector"):
            checkpoint["speaker2dvector"] = self.speaker2dvector
        if hasattr(self, "speaker2priors"):
            checkpoint["speaker2priors"] = self.speaker2priors
        if hasattr(self, "speaker_gmms"):
            checkpoint["speaker_gmms"] = self.speaker_gmms
        if hasattr(self, "dvector_gmms"):
            checkpoint["dvector_gmms"] = self.dvector_gmms

    def forward(self, targets, inference=False):
        # print(
        #     targets["phones"].shape,
        #     targets["mel"].shape,
        #     targets["variances_pitch"].shape,
        #     targets["duration"].shape,
        #     "forward shapes"
        # )
        if not isinstance(targets["duration"], torch.Tensor):
            targets["duration"] = torch.from_numpy(targets["duration"]).to(device="cpu")

        with Timer("phones_and_speaker_generator") as t:
            phones = targets["phones"].to(self.device)
            if self.fastdiff_speaker_generator is None:
                speakers = targets["speaker"].to(self.device)
            else:
                speakers_mean = targets["speaker"].to(self.device)
                if inference:
                    speaker_pred = self.fastdiff_speaker_generator(speakers_mean, inference=True)
                    speakers = speaker_pred
                else:
                    speakers = targets["utterance_dvec"].to(self.device)
                    speaker_pred, speaker_z = self.fastdiff_speaker_generator(speakers_mean, speakers)

            src_mask = phones.eq(0)

            output = self.phone_embedding(phones)

            output = self.positional_encoding(output)

        with Timer("encoder") as t:
            if not self.hparams.speaker_embedding_every_layer:
                output = output + self.speaker_embedding(
                    speakers, output.shape[1], output.shape[-1]
                )
            else:
                every_encoder_layer = self.speaker_embedding(
                    speakers, output.shape[1], output.shape[-1]
                )

            if self.hparams.prior_embedding_every_layer:
                for prior in self.hparams.priors:
                    every_encoder_layer = every_encoder_layer + self.prior_embeddings[
                        prior
                    ](
                        torch.tensor(targets[f"priors_{prior}"]).to(self.device),
                        output.shape[1],
                    )

            if (
                self.hparams.prior_embedding_every_layer
                or self.hparams.speaker_embedding_every_layer
            ):
                output = self.encoder(
                    output,
                    src_key_padding_mask=src_mask,
                    additional_src=every_encoder_layer,
                )
            else:
                output = self.encoder(output, src_key_padding_mask=src_mask)

            if not self.hparams.prior_embedding_every_layer:
                for prior in self.hparams.priors:
                    output = output + self.prior_embeddings[prior](
                        torch.tensor(targets[f"priors_{prior}"]).to(self.device),
                        output.shape[1],
                    )

        with Timer("variances") as t:
            variance_output = self.variance_adaptor(
                output,
                src_mask,
                targets,
                inference=inference,
            )
            output = variance_output["x"]
            output = self.positional_encoding(output)

        with Timer("decoder") as t:
            spk = self.speaker_embedding(
                speakers, output.shape[1], output.shape[-1]
            )

            if self.hparams.speaker_embedding_every_layer:
                output = self.decoder(
                    output,
                    src_key_padding_mask=variance_output["tgt_mask"],
                    additional_src=spk,
                )
            else:
                output = output + spk
                output = self.decoder(
                    output, src_key_padding_mask=variance_output["tgt_mask"]
                )

            output = self.linear(output)

        result = {
            "mel": output,
            "duration_prediction": variance_output["duration_prediction"],
            "duration_rounded": variance_output["duration_rounded"],
            "src_mask": src_mask,
            "tgt_mask": variance_output["tgt_mask"],
        }

        with Timer("fastdiff_vocoder") as t:
            if self.fastdiff_model is not None:
                fd_variance_output = self.fastdiff_linear(variance_output["out"] + spk)
                result["fastdiff_var"] = (
                    fd_variance_output 
                ) * 0.1
            if self.fastdiff_model is not None and not inference:
                if self.current_epoch < self.hparams.fastdiff_schedule_end:
                    sched_idx = self.current_epoch
                else:
                    sched_idx = -1
                if np.random.rand() < self.fastdiff_schedule[sched_idx] or inference:
                    mel_fastdiff = result["mel"] + result["fastdiff_var"]
                    mel_lengths = torch.sum(
                        ~result["tgt_mask"], dim=1, dtype=torch.long
                    )
                    # print(mel_lengths)
                else:
                    mel_fastdiff = targets["mel"] + result["fastdiff_var"]
                    mel_lengths = targets["mel_lengths"]
                # truncate by longest length
                mel_fastdiff = mel_fastdiff[:, :(mel_lengths.max()-2)]
                wav_max_length = (mel_lengths.max()-2)*self.hparams.hop_length
                wav_fastdiff = targets["wav"][:, :wav_max_length]
                result["wav_mask"] = (
                    torch.arange(wav_max_length).to(mel_lengths.device).expand(len(mel_lengths), wav_max_length)
                    < ((mel_lengths-2)*self.hparams.hop_length).unsqueeze(1)
                )
                # the -1 is to avoid edge cases where the mel is longer than the wav
                mel_fastdiff = mel_fastdiff.transpose(1, 2)
                with Timer("fastdiff_vocoder_forward_pass") as t:
                    result["fastdiff"] = self.fastdiff_model(wav_fastdiff, mel_fastdiff, mask=result["wav_mask"])

        for var in self.hparams.variances:
            result[f"variances_{var}"] = variance_output[f"variances_{var}"]
            if self.hparams.fastdiff_variances:
                result[f"variances_{var}_z"] = variance_output[
                    f"variances_{var}_z"
                ]
        
        if self.hparams.fastdiff_variances:
            result["duration_z"] = variance_output["duration_z"]

        if self.fastdiff_speaker_generator is not None:
            if inference:
                result["speaker_z"] = None
            else:
                result["speaker_z"] = speaker_z
            result["speaker_pred"] = speaker_pred

        return result

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        result = self(batch, optimizer_idx)
        losses = self.loss(
            result, batch
        )  # frozen_components=self.variance_adaptor.frozen_components)

        with Timer("log dict") as t:
            if (batch_idx + 1) % 100 == 0:
                log_dict = {f"train/{k}_loss": v.detach() for k, v in losses.items()}
                self.log_dict(
                    log_dict,
                    batch_size=self.batch_size,
                    sync_dist=True,
                )
            return losses["total"]

    def validation_step(self, batch, batch_idx):
        result = self(batch)
        losses = self.loss(result, batch)
        log_dict = {f"eval/{k}_loss": v.detach() for k, v in losses.items()}

        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
        )

        if "xla" not in str(self.device):
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
                print(f"add to results start {self.device}")
                left_to_add = self.valid_nexamples - len(self.eval_log_data)
                inference_result = {k: (v[0:left_to_add] if v is not None else v) for k, v in list(inference_result.items())}
                print("add to results start 2")
                self._add_to_results_dict(inference_result, batch, result)
                print("add to results end")
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
                "duration": batch["duration"][i][~result["src_mask"][i].cpu()],
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
                # try:
                pred_fig = self.valid_ds.plot(pred_dict, show=False)
                true_fig = self.valid_ds.plot(true_dict, show=False)
                if self.valid_example_directory is not None:
                    Path(self.valid_example_directory).mkdir(
                        parents=True, exist_ok=True
                    )
                    pred_fig.save(
                        os.path.join(
                            self.valid_example_directory,
                            f"pred_{batch['id'][i]}.png",
                        )
                    )
                    true_fig.save(
                        os.path.join(
                            self.valid_example_directory,
                            f"true_{batch['id'][i]}.png",
                        )
                    )
                if self.fastdiff_model is None:
                    pred_audio = self.synth(pred_mel.to(self.device).float())[0]
                    true_audio = self.synth(true_mel.to(self.device).float())[0]
                else:
                    pred_mel = pred_mel + inference_result["fastdiff_var"][i][
                        ~inference_result["tgt_mask"][i]
                    ].cpu()
                    true_mel = true_mel + result["fastdiff_var"][i][
                        ~result["tgt_mask"][i]
                    ].cpu()
                    pred_audio = self.fastdiff_model.inference(pred_mel.to(self.device), N=4).cpu()
                    true_audio = self.fastdiff_model.inference(true_mel.to(self.device), N=4).cpu()
                if self.valid_example_directory is not None:
                    torchaudio.save(
                        os.path.join(
                            self.valid_example_directory,
                            f"pred_{batch['id'][i]}.wav",
                        ),
                        pred_audio.unsqueeze(0),
                        22050,
                    )
                    torchaudio.save(
                        os.path.join(
                            self.valid_example_directory,
                            f"true_{batch['id'][i]}.wav",
                        ),
                        true_audio.unsqueeze(0),
                        22050,
                    )
                self.eval_log_data.append(
                    [
                        batch["text"][i],
                        wandb.Image(pred_fig),
                        wandb.Image(true_fig),
                        wandb.Audio(pred_audio, sample_rate=22050),
                        wandb.Audio(true_audio, sample_rate=22050),
                    ]
                )
                # except:
                #     print(
                #         "WARNING: failed to log example (common before training starts)"
                #     )

    def _add_to_results_dict(self, inference_result, batch, result):
        # duration
        self.results_dict["duration"]["pred"] += list(
            inference_result["duration_rounded"][~inference_result["src_mask"]]
        )
        self.results_dict["duration"]["true"] += list(
            batch["duration"][~result["src_mask"].to(batch["duration"].device)]
        )

        # mel
        self.results_dict["mel"]["pred"] += list(
            inference_result["mel"][~inference_result["tgt_mask"]]
        )
        self.results_dict["mel"]["true"] += list(batch["mel"][~result["tgt_mask"]])

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
                )
                self.results_dict[var]["true"] += list(
                    batch[f"variances_{var}_original_signal"][~result[mask]]
                )
            else:
                self.results_dict[var]["pred"] += list(
                    inference_result[f"variances_{var}"][~inference_result[mask]]
                )
                self.results_dict[var]["true"] += list(
                    batch[f"variances_{var}"][~result[mask]]
                )

    # def on_validation_start(self):
    #     for v in self.trainer.val_dataloaders:
    #         v.sampler.shuffle = False
    
    # def on_training_start(self):
    #     for v in self.trainer.train_dataloaders:
    #         v.sampler.shuffle = False

    def validation_epoch_end(self, validation_step_outputs):
        if self.trainer.is_global_zero and "xla" not in str(self.device):
            wandb.init(project="FastSpeech2", settings=wandb.Settings(start_method='thread'))
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
                if key not in "mel":
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
                    try:
                        var_mae = np.mean(
                            np.abs(
                                np.array(self.results_dict[key]["pred"])
                                - np.array(self.results_dict[key]["true"])
                            )
                        )
                    except:
                        self.eval_log_data = None
                        print("WARNING: failed to log validation results, this should be fixed in the next release")
                        return
                    if (
                        self.hparams.variance_early_stopping != "none"
                        and not (
                            key in self.best_variances
                            and self.best_variances[key][1] == -1
                        )
                        and key != "duration"  # TODO: add duration to early stopping
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
                                self.log_dict(
                                    {
                                        f"variance_early_stopping_{key}_epoch": self.current_epoch
                                    }
                                )

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
                    mel_softdtw_1 = SoftDTW(normalize=True, gamma=1)(
                        torch.tensor(self.results_dict[key]["pred"]).float(),
                        torch.tensor(self.results_dict[key]["true"]).float(),
                    )
                    mel_softdtw_0 = SoftDTW(normalize=True, gamma=0.001)(
                        torch.tensor(self.results_dict[key]["pred"]).float(),
                        torch.tensor(self.results_dict[key]["true"]).float(),
                    )
                    self.log_dict(
                        {
                            f"eval/jensenshannon_{key}": mel_js,
                            f"eval/softdtw_{key}": mel_softdtw_1,
                            f"eval/softdtw_{key}_gamma0": mel_softdtw_0,
                        }
                    )
            self.eval_log_data = None

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
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
        parser.add_argument("--layer_dropout", type=float, default=0.1)
        parser.add_argument(
            "--variances", nargs="+", type=str, default=["pitch", "energy", "snr"]
        )
        parser.add_argument(
            "--variance_levels",
            nargs="+",
            type=str,
            default=["frame", "frame", "frame", "frame"],
        )
        parser.add_argument(
            "--variance_transforms",
            nargs="+",
            type=str,
            default=["cwt", "none", "none", "none"],
        )
        parser.add_argument(
            "--variance_losses",
            nargs="+",
            type=str,
            default=["mse", "mse", "mse", "mse"],
        )
        parser.add_argument(
            "--variance_nlayers", nargs="+", type=int, default=[5, 5, 5, 5]
        )
        parser.add_argument(
            "--variance_loss_weights", nargs="+", type=float, default=[1, 1e-1, 1e-1, 1e-1]
        )
        parser.add_argument(
            "--variance_kernel_size", nargs="+", type=int, default=[3, 3, 3, 3]
        )
        parser.add_argument(
            "--variance_dropout", nargs="+", type=float, default=[0.5, 0.5, 0.5, 0.5]
        )
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
        parser.add_argument("--duration_loss", type=str, default="mse")
        parser.add_argument("--mel_loss", type=str, default="l1")
        parser.add_argument("--soft_dtw_gamma", type=float, default=0.1)
        parser.add_argument("--soft_dtw_chunk_size", type=int, default=256)
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
        parser.add_argument(
            "--encoder_kernel_sizes", nargs="+", type=int, default=[5, 25, 13, 9]
        )
        parser.add_argument("--encoder_dim_feedforward", type=int, default=None)
        parser.add_argument("--encoder_conformer", type=str2bool, default=True)
        parser.add_argument("--encoder_depthwise_conv", type=str2bool, default=True)
        parser.add_argument("--encoder_conv_filter_size", type=int, default=1024)
        parser.add_argument("--decoder_hidden", type=int, default=256)
        parser.add_argument("--decoder_head", type=int, default=2)
        parser.add_argument("--decoder_layers", type=int, default=4)
        parser.add_argument("--decoder_dropout", type=float, default=0.1)
        parser.add_argument(
            "--decoder_kernel_sizes", nargs="+", type=int, default=[17, 21, 9, 13]
        )
        parser.add_argument("--decoder_dim_feedforward", type=int, default=None)
        parser.add_argument("--decoder_conformer", type=str2bool, default=True)
        parser.add_argument("--decoder_depthwise_conv", type=str2bool, default=True)
        parser.add_argument("--decoder_conv_filter_size", type=int, default=1024)
        parser.add_argument("--valid_nexamples", type=int, default=10)
        parser.add_argument("--valid_example_directory", type=str, default=None)
        parser.add_argument("--variance_early_stopping", type=str, default="none")
        parser.add_argument("--variance_early_stopping_patience", type=int, default=4)
        parser.add_argument(
            "--variance_early_stopping_directory", type=str, default="variance_encoders"
        )
        parser.add_argument("--num_workers", type=int, default=num_cpus)
        parser.add_argument(
            "--speaker_embedding_every_layer", type=str2bool, default=False
        )
        parser.add_argument(
            "--prior_embedding_every_layer", type=str2bool, default=False
        )
        # priors gmm 
        parser.add_argument("--priors_gmm", type=str2bool, default=False)
        parser.add_argument("--priors_gmm_max_components", type=int, default=5)
        parser.add_argument("--priors_gmm_min_samples_per_component", type=int, default=20)
        parser.add_argument("--priors_gmm_reg_covar", type=float, default=1e-3)
        parser.add_argument("--priors_gmm_logs", nargs="+", type=int, default=[0,1,2,3])
        parser.add_argument("--dvector_gmm", type=str2bool, default=False)
        parser.add_argument("--fastdiff_schedule", nargs="+", type=int, default=[0.1,1])
        parser.add_argument("--fastdiff_schedule_start", type=int, default=0)
        parser.add_argument("--fastdiff_schedule_end", type=int, default=30)
        parser.add_argument("--fastdiff_variances", type=str2bool, default=False)
        parser.add_argument("--fastdiff_speakers", type=str2bool, default=False)
        parser.add_argument("--sort_data_by_length", type=str2bool, default=False)
        return parent_parser

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parent_parser = TTSDataset.add_model_specific_args(parent_parser, "train")
        parser = parent_parser.add_argument_group("Valid Dataset")
        parser.add_argument("--valid_max_entries", type=int, default=None)
        parser.add_argument("--valid_shuffle_seed", type=int, default=42)
        return parent_parser

    def train_dataloader(self):
        if self.hparams.sort_data_by_length:
            self.train_ds.batch_by_duration(True, bins=2)
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.train_ds._collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.hparams.sort_data_by_length:
            self.valid_ds.batch_by_duration(True, bins=1)
        return DataLoader(
            self.valid_ds,
            batch_size=10, # TODO: make this a parameter
            collate_fn=self.valid_ds._collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )
