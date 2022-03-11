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

wandb_logger = WandbLogger(project="LightningFastSpeech",)

if __name__ == "__main__":
    epochs = config["train"].getint("epochs")
    validation_step = config["train"].getint("validation_step")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # model = FastSpeech2(
    #     learning_rate=config["train"].getfloat("lr"),
    # )
    model = FastSpeech2.load_from_checkpoint('models/31_epochs.ckpt')
    trainer = Trainer(
        default_root_dir="logs",
        min_epochs=epochs,
        max_epochs=epochs,
        val_check_interval=validation_step,
        logger=wandb_logger,
        accumulate_grad_batches=config["train"].getint("gradient_accumulation"),
        callbacks=[lr_monitor],
        gradient_clip_val=config["train"].getint("gradient_clipping"),
        gpus=-1,
        precision=16,
        strategy="dp",
        #auto_select_gpus=True,
    )
    trainer.fit(model)
