import json
import torch
import hifigan

class Synthesiser():

    def __init__(self, sampling_rate):
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config["sampling_rate"] = sampling_rate
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to("cuda:0")
        self.vocoder = vocoder

    def __call__(self, mel):
        mel = torch.unsqueeze(mel.T, 0)
        return (self.vocoder(mel.to("cuda:0")).squeeze(1).cpu().detach().numpy() * 32768.0).astype("int16")