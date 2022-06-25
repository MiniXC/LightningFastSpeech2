from argparse import ArgumentParser

import torch
import torch.multiprocessing

from fastspeech2.fastspeech2 import FastSpeech2
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import os

from alignments.datasets.libritts import LibrittsDataset

os.environ["WANDB_MODE"] = "offline"

torch.multiprocessing.set_sharing_strategy("file_system")

wandb_logger = WandbLogger(project="LightningFastSpeech")

if __name__ == "__main__":
    epochs = 1
    validation_step = 1.0
    lr_monitor = LearningRateMonitor(logging_interval="step")

    model = FastSpeech2(
        LibrittsDataset(
            "../data/dev-clean-aligned",
            "../data/dev-clean",
            "https://www.openslr.org/resources/60/dev-clean.tar.gz",
            verbose=True,
        ),
        LibrittsDataset(
            "../data/train-clean-aligned",
            "../data/train-clean",
            "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
            verbose=True,
        ),
        valid_example_directory="examples"
    )

    strategy = None # "ddp_find_unused_parameters_false"
    gpus = 1

    trainer = Trainer(
        accelerator="gpu",
        # precision=16,
        default_root_dir="logs",
        min_epochs=epochs,
        max_epochs=epochs,
        val_check_interval=validation_step,
        logger=wandb_logger,
        accumulate_grad_batches=6,
        callbacks=[lr_monitor],
        gpus=gpus,
        strategy=strategy,
    )
    # TODO: check gradient clipping effect
    trainer.fit(model)

# accelerator="cpu",
#         detect_anomaly=True,