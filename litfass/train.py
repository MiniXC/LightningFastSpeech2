"""
Script used for training the model.
"""

from argparse import ArgumentParser
import os
import inspect
from pathlib import Path
import json
import hashlib
import pickle

import torch
import torch.multiprocessing
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from alignments.datasets.libritts import LibrittsDataset

from litfass.third_party.argutils import str2bool
from litfass.fastspeech2.fastspeech2 import FastSpeech2
from litfass.third_party.fastdiff.FastDiff import FastDiff

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--early_stopping", type=str2bool, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=4)

    parser.add_argument("--swa_lr", type=float, default=None)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]

    parser.add_argument("--dataset_cache_path", type=str, default="../dataset_cache")
    parser.add_argument("--no_cache", type=str2bool, default=False)

    parser.add_argument(
        "--train_target_path",
        type=str,
        nargs="+",
        default=["../data/train-clean-360-aligned"],
    )
    parser.add_argument(
        "--train_source_path", type=str, nargs="+", default=["../data/train-clean-360"]
    )
    parser.add_argument(
        "--train_source_url",
        type=str,
        nargs="+",
        default=["https://www.openslr.org/resources/60/train-clean-360.tar.gz"],
    )
    parser.add_argument("--train_tmp_path", type=str, default="../tmp")

    parser.add_argument(
        "--valid_target_path", type=str, default="../data/dev-clean-aligned"
    )
    parser.add_argument("--valid_source_path", type=str, default="../data/dev-clean")
    parser.add_argument(
        "--valid_source_url",
        type=str,
        default="https://www.openslr.org/resources/60/dev-clean.tar.gz",
    )
    parser.add_argument("--valid_tmp_path", type=str, default="../tmp")

    parser = FastSpeech2.add_model_specific_args(parser)
    parser = FastSpeech2.add_dataset_specific_args(parser)

    parser.add_argument("--wandb_project", type=str, default="fastspeech2")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str2bool, default=True)
    parser.add_argument("--checkpoint_key", type=str, default="eval/mel_loss")
    parser.add_argument("--checkpoint_mode", type=str, default="min")
    parser.add_argument("--checkpoint_path", type=str, default="models")
    parser.add_argument("--checkpoint_filename", type=str, default=None)
    parser.add_argument("--from_checkpoint", type=str, default=None)

    parser.add_argument("--visible_gpus", type=int, default=0)

    # fastdiff vocoder
    parser.add_argument("--fastdiff_vocoder", type=str2bool, default=False)
    parser.add_argument("--fastdiff_vocoder_checkpoint", type=str, default=None)
    parser = FastDiff.add_model_specific_args(parser)

    args = parser.parse_args()
    var_args = vars(args)

    os.environ["WANDB_MODE"] = var_args["wandb_mode"]
    os.environ["WANDB_WATCH"] = "false"
    if var_args["wandb_name"] is None:
        wandb_logger = WandbLogger(project=var_args["wandb_project"])
    else:
        wandb_logger = WandbLogger(
            project=var_args["wandb_project"], name=var_args["wandb_name"]
        )

    train_ds = []

    train_ds_kwargs = {
        k.replace("train_", ""): v
        for k, v in var_args.items()
        if k.startswith("train_")
    }

    valid_ds_kwargs = {
        k.replace("valid_", ""): v
        for k, v in var_args.items()
        if k.startswith("valid_")
    }

    if var_args["fastdiff_vocoder"]:
        train_ds_kwargs["load_wav"] = True
        valid_ds_kwargs["load_wav"] = True
        fastdiff_args = {
            k.replace("fastdiff_", ""): v
            for k, v in var_args.items()
            if (
                k.startswith("fastdiff_") and
                 "schedule" not in k and 
                 "vocoder" not in k and
                 "variances" not in k and
                 "speaker" not in k
            )
        }
        fastdiff_model = FastDiff(**fastdiff_args)
        if var_args["fastdiff_vocoder_checkpoint"] is not None:
            state_dict = torch.load(var_args["fastdiff_vocoder_checkpoint"])["state_dict"]["model"]
            fastdiff_model.load_state_dict(state_dict, strict=True)
    else:
        fastdiff_model = None

    if not var_args["no_cache"]:
        Path(var_args["dataset_cache_path"]).mkdir(parents=True, exist_ok=True)
        cache_path = Path(var_args["dataset_cache_path"])
    else:
        cache_path = None

    for i in range(len(var_args["train_target_path"])):
        if not var_args["no_cache"]:
            kwargs = train_ds_kwargs
            kwargs.update({"target_directory": var_args["train_target_path"][i]})
            ds_hash = hashlib.md5(
                json.dumps(kwargs, sort_keys=True).encode("utf-8")
            ).hexdigest()
            cache_path_alignments = (
                Path(var_args["dataset_cache_path"]) / f"train-alignments-{ds_hash}.pt"
            )
        if (
            var_args["no_cache"]
            or next(Path(var_args["train_target_path"][i]).rglob("**/*.TextGrid"), -1)
            == -1
            or not cache_path_alignments.exists()
        ):
            if len(var_args["train_source_path"]) > i:
                source_path = var_args["train_source_path"][i]
            else:
                source_path = None
            if len(var_args["train_source_url"]) > i:
                source_url = var_args["train_source_url"][i]
            else:
                source_url = None
            train_ds += [
                LibrittsDataset(
                    target_directory=var_args["train_target_path"][i],
                    source_directory=source_path,
                    source_url=source_url,
                    verbose=True,
                    tmp_directory=var_args["train_tmp_path"],
                    chunk_size=10_000,
                )
            ]
            if not var_args["no_cache"]:
                train_ds[-1].hash = ds_hash
                with open(cache_path_alignments, "wb") as f:
                    pickle.dump(train_ds[-1], f)
        else:
            if cache_path_alignments.exists():
                with open(cache_path_alignments, "rb") as f:
                    train_ds += [pickle.load(f)]

    if not var_args["no_cache"]:
        kwargs = valid_ds_kwargs
        kwargs.update({"target_directory": var_args["valid_target_path"]})
        ds_hash = hashlib.md5(
            json.dumps(kwargs, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path_alignments = (
            Path(var_args["dataset_cache_path"]) / f"valid-alignments-{ds_hash}.pt"
        )
    if (
        var_args["no_cache"]
        or next(Path(var_args["valid_target_path"]).rglob("**/*.TextGrid"),-1) == -1
        or not cache_path_alignments.exists()
    ):
        valid_ds = LibrittsDataset(
            target_directory=var_args["valid_target_path"],
            source_directory=var_args["valid_source_path"],
            source_url=var_args["valid_source_url"],
            verbose=True,
            tmp_directory=var_args["valid_tmp_path"],
            chunk_size=10_000,
        )
        if not var_args["no_cache"]:
            valid_ds.hash = ds_hash
            with open(cache_path_alignments, "wb") as f:
                pickle.dump(valid_ds, f)
    else:
        if cache_path_alignments.exists():
            with open(cache_path_alignments, "rb") as f:
                valid_ds = pickle.load(f)

    model_args = {
        k: v
        for k, v in var_args.items()
        if k in inspect.signature(FastSpeech2).parameters
    }

    del train_ds_kwargs["target_path"]
    del train_ds_kwargs["target_directory"]
    del train_ds_kwargs["source_path"]
    del train_ds_kwargs["source_url"]
    del train_ds_kwargs["tmp_path"]
    del valid_ds_kwargs["target_path"]
    del valid_ds_kwargs["target_directory"]
    del valid_ds_kwargs["source_path"]
    del valid_ds_kwargs["source_url"]
    del valid_ds_kwargs["nexamples"]
    del valid_ds_kwargs["example_directory"]
    del valid_ds_kwargs["tmp_path"]
    if "load_wav" in valid_ds_kwargs:
        del valid_ds_kwargs["load_wav"]

    if args.from_checkpoint is not None:
        model = FastSpeech2.load_from_checkpoint(
            args.from_checkpoint,
            train_ds=train_ds,
            valid_ds=valid_ds,
            train_ds_kwargs=train_ds_kwargs,
            valid_ds_kwargs=valid_ds_kwargs,
            strict=False,
            fastdiff_model=fastdiff_model,
            **model_args,
        )
    else:
        model_args["cache_path"] = cache_path
        model = FastSpeech2(
            train_ds,
            valid_ds,
            train_ds_kwargs=train_ds_kwargs,
            valid_ds_kwargs=valid_ds_kwargs,
            fastdiff_model=fastdiff_model,
            **model_args,
        )

    if var_args["checkpoint_filename"] is None and var_args["wandb_name"] is not None:
        var_args["checkpoint_filename"] = var_args["wandb_name"]

    if var_args["checkpoint"]:
        callbacks.append(
            ModelCheckpoint(
                monitor=var_args["checkpoint_key"],
                mode=var_args["checkpoint_mode"],
                filename=var_args["checkpoint_filename"],
                dirpath=var_args["checkpoint_path"],
            )
        )

    if var_args["early_stopping"]:
        callbacks.append(
            EarlyStopping(
                monitor="eval/mel_loss", patience=var_args["early_stopping_patience"]
            )
        )

    if var_args["swa_lr"] is not None:
        callbacks.append(StochasticWeightAveraging(swa_lrs=var_args["swa_lr"]))

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        default_root_dir="logs",
        #logger=wandb_logger,
    )

    trainer.fit(model)
