import configparser
from fastspeech2 import FastSpeech2
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project='LightningFastSpeech')

config = configparser.ConfigParser()
config.read("config.ini")

if __name__ == "__main__":
    epochs = config["train"].getint("epochs")
    validation_step = config["train"].getint("validation_step")
    model = FastSpeech2()
    trainer = Trainer(
        gpus=1,
        min_epochs=epochs,
        max_epochs=epochs,
        val_check_interval=validation_step,
        logger=wandb_logger,
    )
    trainer.fit(model)
