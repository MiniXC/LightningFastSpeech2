from argparse import ArgumentParser

import torch
import torch.multiprocessing

from fastspeech2.fastspeech2 import FastSpeech2
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import os
import matplotlib.pyplot as plt

from alignments.datasets.libritts import LibrittsDataset

os.environ["WANDB_MODE"] = "offline"

torch.multiprocessing.set_sharing_strategy("file_system")

wandb_logger = WandbLogger(project="FastSpeech2")

if __name__ == "__main__":
    epochs = 25
    validation_step = 1.0
    lr_monitor = LearningRateMonitor(logging_interval="step")

    train_ds = LibrittsDataset(
        "../data/train-clean-aligned",
        "../data/train-clean",
        "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
        verbose=True,
    )

    valid_ds = LibrittsDataset(
        "../data/dev-clean-aligned",
        "../data/dev-clean",
        "https://www.openslr.org/resources/60/dev-clean.tar.gz",
        verbose=True,
    )

    model = FastSpeech2(
        train_ds,
        valid_ds,
        valid_example_directory="examples",
        batch_size=12,
    )

    strategy = None # "ddp_find_unused_parameters_false"
    gpus = 1

    trainer = Trainer(
        accelerator="gpu",
        precision=16,
        default_root_dir="logs",
        min_epochs=epochs,
        max_epochs=epochs,
        val_check_interval=validation_step,
        logger=wandb_logger,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        callbacks=[lr_monitor],
        gpus=gpus,
        strategy=strategy,
    )
    # TODO: check gradient clipping effect
    trainer.fit(model)