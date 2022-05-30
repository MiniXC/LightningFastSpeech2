import json
import torch
import hifigan


class Synthesiser:
    def __init__(self, sampling_rate, device="cuda:0"):
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config["sampling_rate"] = sampling_rate
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        ckpt = torch.load("hifigan/generator_universal.pth.tar")
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
