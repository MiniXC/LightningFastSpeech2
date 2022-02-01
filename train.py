import configparser
from fastspeech2 import FastSpeech2
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import os

# os.environ["WANDB_MODE"] = "offline"

wandb_logger = WandbLogger(project="LightningFastSpeech")

config = configparser.ConfigParser()
config.read("config.ini")

if __name__ == "__main__":
    epochs = config["train"].getint("epochs")
    validation_step = config["train"].getint("validation_step")
    model = FastSpeech2.load_from_checkpoint(
        "logs/None/version_None/checkpoints/epoch=9-step=2890-v1.ckpt"
    )
    trainer = Trainer(
        default_root_dir="logs",
        gpus=1,
        min_epochs=epochs,
        max_epochs=epochs,
        val_check_interval=validation_step,
        logger=wandb_logger,
    )
    trainer.fit(model)
