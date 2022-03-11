import configparser
from fastspeech2 import FastSpeech2
from synthesiser import Synthesiser
import os
import json
import torchaudio
import torch
import numpy as np

config = configparser.ConfigParser()
config.read("synth.ini")

if __name__ == "__main__":
    model = FastSpeech2.load_from_checkpoint(config['model'].get('path')).to("cuda:0")
    synthesiser = Synthesiser(config['model'].getint('sampling_rate'))
    for batch in model.val_dataloader():
        #print(batch)
        preds, src_mask, tgt_mask = model(batch["phones"], batch["speaker"])
        i = 0
        pitch, energy, duration = preds[1], preds[2], preds[4]
        
        duration = duration.float().cpu()
        duration *= np.random.uniform(.5, 1.5, size=duration.shape)
        duration = torch.round(duration).int()
        
        pitch = pitch.float().detach().cpu()
        pitch *= np.random.uniform(.5, 1.5, size=pitch.shape)
        
        energy = energy.float().detach().cpu()
        energy *= np.random.uniform(.5, 1.5, size=energy.shape)

        preds, src_mask, tgt_mask = model(batch["phones"], batch["speaker"], pitch, energy, duration)
        audio = synthesiser(preds[0][i][~tgt_mask[i]])
        torchaudio.save('test.wav', torch.tensor(audio), config['model'].getint('sampling_rate'))
        break
    #preds, src_mask, tgt_mask = self(batch["phones"], batch["speaker"])

# 1
# duration = duration.float() * 1.2
# duration = torch.round(duration).int()

# 2
# duration = duration.float()
# dur_mean = duration.mean()
# duration = (duration - dur_mean) * 1.2
# duration = torch.clamp(torch.round(duration + dur_mean), min=0).int()