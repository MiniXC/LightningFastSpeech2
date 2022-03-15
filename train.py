import configparser
from fastspeech2 import FastSpeech2
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import os

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

config = configparser.ConfigParser()
config.read("config.ini")

os.environ["WANDB_MODE"] = config["train"].get("wandb_mode")

wandb_logger = WandbLogger(project="LightningFastSpeech", group="DDP", log_model="all")

if __name__ == "__main__":
    epochs = config["train"].getint("epochs")
    validation_step = config["train"].getfloat("validation_step")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model_path = config["train"].get('model_path')
    if model_path == "None":
        model = FastSpeech2(
            learning_rate=config["train"].getfloat("lr"),
        )
    else:
        model = FastSpeech2.load_from_checkpoint(model_path, strict=False)
        
    if config["train"].getboolean('distributed'):
        strategy = "ddp"
        gpus = -1
    else:
        strategy = None
        gpus = 1

    trainer = Trainer(
        default_root_dir="logs",
        min_epochs=epochs,
        max_epochs=epochs,
        val_check_interval=validation_step,
        logger=wandb_logger,
        accumulate_grad_batches=config["train"].getint("gradient_accumulation"),
        callbacks=[lr_monitor],
        gradient_clip_val=config["train"].getint("gradient_clipping"),
        gpus=gpus,
        strategy=strategy,
    )
    trainer.fit(model)
