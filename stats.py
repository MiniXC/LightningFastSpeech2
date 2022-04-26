import configparser
from fastspeech2 import FastSpeech2
from synthesiser import Synthesiser
import os
import json
import torchaudio
import torch
import numpy as np
import click
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

config = configparser.ConfigParser()
config.read("synth.ini")

diversity_choice = click.Choice(['increase', 'decrease'])

def segment1d(x):
    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(x.reshape(-1, 1))
    s = np.linspace(x.min()*1.1, x.max()*1.1)
    e = kde.score_samples(s.reshape(-1, 1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    result = []
    for entry in x:
        less = float(entry)<mi
        if any(less):
            result.append(float(ma[np.argmax(less)]))
        else:
            result.append(ma[-1])
    return torch.Tensor(result)

def remove_outliers(x):
    x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    x = x[x > mean - std * 3]
    x = x[x < mean + std * 3]
    return x

@click.command()
@click.argument('key', type=str)
def stats(key):
    model = FastSpeech2.load_from_checkpoint(config['model'].get('path')).to("cuda:0")

    synth_durations = []
    real_durations = []

    i = 0

    for batch in tqdm(model.train_dataloader(), total=100):
        
        preds, src_mask, tgt_mask = model(batch["phones"], batch["speaker"])
        pitch, energy, duration = preds[1], preds[2], preds[4]

        synth_var = {
            'pitch':pitch,
            'energy':energy,
            'duration':duration,
        }

        synth_durations += [float(x) for x in synth_var[key][synth_var[key]!=0]]
        real_durations += [float(x) for x in batch[key][batch[key]!=0]]

        i += 1

        if i > 100:
            break

    #synth_durations = remove_outliers(synth_durations)
    #real_durations = remove_outliers(real_durations)

    sns.displot(x=list(synth_durations) + list(real_durations), hue=["synthesised"] * len(synth_durations) + ['real'] * len(real_durations), kde=True)
    #sns.boxplot(x=list(synth_durations) + list(real_durations), y=["synthesised"] * len(synth_durations) + ['real'] * len(real_durations))
    plt.title(f'{key} distribution')
    plt.show()

if __name__ == "__main__":
    stats()

# 1
# duration = duration.float() * 1.2
# duration = torch.round(duration).int()

# 2
# duration = duration.float()
# dur_mean = duration.mean()
# duration = (duration - dur_mean) * 1.2
# duration = torch.clamp(torch.round(duration + dur_mean), min=0).int()
