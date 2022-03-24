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

@click.command()
@click.argument('destination', type=str)
@click.argument('size', type=float)
@click.option('--pitch_diversity', type=diversity_choice)
@click.option('--energy_diversity', type=diversity_choice)
@click.option('--duration_diversity', type=diversity_choice)
@click.option('--speaker_diversity', type=diversity_choice)
@click.option('--lexical_diversity', type=diversity_choice)
@click.option('--copy', is_flag=True)
def synth(
    destination,
    size,
    pitch_diversity=None,
    energy_diversity=None,
    duration_diversity=None,
    speaker_diversity=None,
    lexical_diversity=None,
    copy=False,
):
    try:
        shutil.rmtree(destination)
    except FileNotFoundError:
        pass
    Path(destination).mkdir(exist_ok=True)

    model = FastSpeech2.load_from_checkpoint(config['model'].get('path')).to("cuda:0")
    synthesiser = Synthesiser(config['model'].getint('sampling_rate'), device='cuda:1')

    total_len = 0
    max_len = size * 60 * 60

    if speaker_diversity is not None:
        speakers = list(model.train_ds.speaker_map.values())
        if speaker_diversity == 'decrease':
            single_speaker = speakers[np.random.randint(0, len(speakers))]
            print(f'only using speaker {single_speaker}')

    for batch in tqdm(model.train_dataloader()):
        if copy:
            for i in range(len(batch['speaker'])):
                wav_file = glob(os.path.join('../Data/LibriTTS/train-clean-100-aligned','**',batch["id"][i]), recursive=True)[0]
                lab_file = wav_file.replace(".wav", ".lab")
                speaker_str = batch['id'][i].split('_')[0]
                Path(os.path.join(destination,speaker_str)).mkdir(  exist_ok=True)
                shutil.copyfile(wav_file, os.path.join(destination,speaker_str,batch['id'][i]))
                shutil.copyfile(lab_file, os.path.join(destination,speaker_str,batch['id'][i].replace(".wav", ".lab")))
                audio, sampling_rate = torchaudio.load(wav_file)
                total_len += len(audio[0]) / sampling_rate
                print(f'{round(total_len / 60 / 60, 3)} / {round(max_len / 60 / 60, 3)} h')

            if total_len >= max_len:
                break
            continue

        preds, src_mask, tgt_mask = model(batch["phones"], batch["speaker"])
        i = 0
        pitch, energy, duration = preds[1], preds[2], preds[4]
        
        if duration_diversity is not None:
            duration = duration.float().cpu()
            if duration_diversity == 'increase':
                duration *= np.random.uniform(.5, 1.5, size=duration.shape)
                duration = torch.round(duration).int()
            if duration_diversity == 'decrease':
                duration[~src_mask] = segment1d(duration[~src_mask])
                duration = duration.int()
        
        if pitch_diversity is not None:
            pitch = pitch.float().detach().cpu()
            if pitch_diversity == 'increase':
                pitch *= np.random.uniform(0.5, 1.5, size=pitch.shape)
            if pitch_diversity == 'decrease':  
                pitch[~src_mask] = segment1d(pitch[~src_mask])
        
        if energy_diversity is not None:
            energy = energy.float().detach().cpu()
            if energy_diversity == 'increase':
                energy *= np.random.uniform(0.5, 1.5)
            if energy_diversity == 'decrease':
                energy[~src_mask] = segment1d(energy[~src_mask])

        if lexical_diversity is not None:
            raise NotImplementedError()

        if speaker_diversity == 'increase':
            batch['speaker'] = torch.Tensor([speakers[np.random.randint(0, len(speakers))] for _ in batch['speaker']]).int()
        if speaker_diversity == 'decrease':
            batch['speaker'] = torch.Tensor([single_speaker for _ in batch['speaker']]).int()

        preds, src_mask, tgt_mask = model(batch["phones"], batch["speaker"], pitch, energy, duration)

        for i in range(len(batch['speaker'])):

            audio = synthesiser(preds[0][i][~tgt_mask[i]])
            total_len += audio.shape[1] / float(config['model'].getint('sampling_rate'))

            speaker_str = batch['id'][i].split('_')[0]
            Path(os.path.join(destination,speaker_str)).mkdir(exist_ok=True)

            torchaudio.save(
                os.path.join(destination,speaker_str,f'{batch["id"][i]}'),
                torch.tensor(audio),
                config['model'].getint('sampling_rate'),
            )
            with open(os.path.join(destination,speaker_str,f'{batch["id"][i]}').replace(".wav",".lab"), 'w') as lab:
                lab.write(batch["text"][i])

        print(f'{round(total_len / 60 / 60, 3)} / {round(max_len / 60 / 60, 3)} h')

        if total_len >= max_len:
            break

if __name__ == "__main__":
    synth()

# 1
# duration = duration.float() * 1.2
# duration = torch.round(duration).int()

# 2
# duration = duration.float()
# dur_mean = duration.mean()
# duration = (duration - dur_mean) * 1.2
# duration = torch.clamp(torch.round(duration + dur_mean), min=0).int()
