import json
import torch
from .models import Generator

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# TODO: decompress dynamic range

class Synthesiser:
    def __init__(
        self,
        device="cuda:0",
        model="universal",
    ):
        with open("third_party/hifigan/config.json", "r") as f:
            config = json.load(f)
        config = AttrDict(config)
        vocoder = Generator(config)
        ckpt = torch.load(f"third_party/hifigan/generator_{model}.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        self.device = device
        vocoder.to(self.device)
        self.vocoder = vocoder

    def __call__(self, mel):
        mel = torch.unsqueeze(mel.T, 0)
        result = (
            self.vocoder(mel.to(self.device)).squeeze(1).cpu().detach().numpy()
            * 32768.0
        ).astype("int16")
        return result
