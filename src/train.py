from argparse import ArgumentParser
import os
import inspect

import torch
import torch.multiprocessing

from fastspeech2.fastspeech2 import FastSpeech2
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt

from alignments.datasets.libritts import LibrittsDataset

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]

    parser.add_argument("--train_target_path", type=str, default="../data/train-clean-aligned")
    parser.add_argument("--train_source_path", type=str, default="../data/train-clean")
    parser.add_argument("--train_source_url", type=str, default="https://www.openslr.org/resources/60/train-clean-100.tar.gz")

    parser.add_argument("--valid_target_path", type=str, default="../data/dev-clean-aligned")
    parser.add_argument("--valid_source_path", type=str, default="../data/dev-clean")
    parser.add_argument("--valid_source_url", type=str, default="https://www.openslr.org/resources/60/dev-clean.tar.gz")

    parser = FastSpeech2.add_model_specific_args(parser)
    parser = FastSpeech2.add_dataset_specific_args(parser)

    parser.add_argument("--wandb_project", type=str, default="fastspeech2")
    parser.add_argument("--wandb_mode", type=str, default="online")

    parser.add_argument("--visible_gpus", type=int, default=0)

    args = parser.parse_args()
    var_args = vars(args)

    os.environ["WANDB_MODE"] = var_args["wandb_mode"]
    wandb_logger = WandbLogger(project=var_args["wandb_project"])

    train_ds = LibrittsDataset(
        var_args["train_target_path"],
        var_args["train_source_path"],
        var_args["train_source_url"],
        verbose=True,
    )

    valid_ds = LibrittsDataset(
        var_args["valid_target_path"],
        var_args["valid_source_path"],
        var_args["valid_source_url"],
        verbose=True,
    )

    model_args = {k: v for k, v in var_args.items() if k in inspect.signature(FastSpeech2).parameters}

    del var_args["train_target_path"]
    del var_args["train_source_path"]
    del var_args["train_source_url"]
    del var_args["valid_target_path"]
    del var_args["valid_source_path"]
    del var_args["valid_source_url"]
    del var_args["valid_nexamples"]
    del var_args["valid_example_directory"]

    model = FastSpeech2(
        train_ds,
        valid_ds,
        train_ds_kwargs={k.replace("train_", ""): v for k, v in var_args.items() if k.startswith("train_")},
        valid_ds_kwargs={k.replace("valid_", ""): v for k, v in var_args.items() if k.startswith("valid_")},
        **model_args,
    )

    if var_args["early_stopping"]:
        callbacks.append(EarlyStopping(monitor="eval/total_loss", patience=var_args["early_stopping_patience"]))

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        default_root_dir="logs",
        logger=wandb_logger,
        strategy=None,
    )

    trainer.fit(model)