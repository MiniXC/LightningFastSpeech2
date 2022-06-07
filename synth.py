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
from snr import wada_snr
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from dataset import UnprocessedDataset, ProcessedDataset
import random
from torch.utils.data import DataLoader
from torch.multiprocessing import Pool
from scipy.io.wavfile import write
import pandas as pd

config = configparser.ConfigParser()
config.read("synth.ini")

diversity_choice = click.Choice(["increase", "decrease"])


def int16_samples_to_float32(y):
    """Convert int16 numpy array of audio samples to float32."""
    if y.dtype != np.int16:
        if y.dtype == np.float32:
            return y
        elif y.dtype == torch.float32:
            return y.numpy()
        else:
            raise ValueError(f"input samples not int16 or float32, but {y.dtype}")
    return y.astype(np.float32) / np.iinfo(np.int16).max


def float_samples_to_int16(y):
    """Convert floating-point numpy array of audio samples to int16."""
    if not issubclass(y.dtype.type, np.floating):
        raise ValueError("input samples not floating-point")
    try:
        result = (y * np.iinfo(np.int16).max).astype(np.int16)
    except Warning:
        print(
            y, np.iinfo(np.int16).max
        )  # sometimes catch warnings that limit has been exceeded
        raise
    return result


def segment1d(x):
    kde = KernelDensity(kernel="gaussian", bandwidth=1).fit(x.reshape(-1, 1))
    s = np.linspace(x.min() * 1.1, x.max() * 1.1)
    e = kde.score_samples(s.reshape(-1, 1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    result = []
    for entry in x:
        less = float(entry) < mi
        if any(less):
            result.append(float(ma[np.argmax(less)]))
        else:
            result.append(ma[-1])
    return torch.Tensor(result)


def copy_reference(batch_id):
    augmentation = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        ]
    )
    speaker_str = batch_id.split("_")[0]
    wav_file = glob(
        os.path.join("../Data/LibriTTS/train-clean-360-aligned", "**", batch_id),
        recursive=True,
    )[0]
    lab_file = wav_file.replace(".wav", ".lab")
    Path(os.path.join(g_destination, speaker_str)).mkdir(exist_ok=True)
    audio, sr = torchaudio.load(wav_file)
    if g_augment:
        audio = augmentation(int16_samples_to_float32(audio), sample_rate=sr)
        audio = float_samples_to_int16(audio)
    write(os.path.join(g_destination, speaker_str, batch_id), sr, audio[0])
    shutil.copyfile(
        lab_file,
        os.path.join(g_destination, speaker_str, batch_id.replace(".wav", ".lab")),
    )
    return len(audio[0]) / sr


@click.command()
@click.argument("destination", type=str)
@click.argument("size", type=float)
@click.option("--pitch_diversity", type=diversity_choice)
@click.option("--energy_diversity", type=diversity_choice)
@click.option("--duration_diversity", type=diversity_choice)
@click.option("--speaker_diversity", type=diversity_choice)
@click.option("--lexical_diversity", type=diversity_choice)
@click.option("--copy", is_flag=True)
@click.option("--resynthesise", is_flag=True)
@click.option("--oracle", type=str)
@click.option("--snr-threshold", type=float)
@click.option("--augment", is_flag=True)
@click.option("--num_speakers")
@click.option("--duration_augment")
def synth(
    destination,
    size,
    pitch_diversity=None,
    energy_diversity=None,
    duration_diversity=None,
    speaker_diversity=None,
    lexical_diversity=None,
    copy=False,
    resynthesise=False,
    oracle=None,
    snr_threshold=None,
    augment=False,
    num_speakers=None,
    duration_augment=None,
):
    global g_destination, g_augment, augmentation

    g_destination, g_augment = destination, augment

    try:
        shutil.rmtree(destination)
    except FileNotFoundError:
        pass
    Path(destination).mkdir(exist_ok=True, parents=True)

    train_orig = UnprocessedDataset(
        "../Data/LibriTTS/train-clean-360-aligned",
    )
    train_orig_p = ProcessedDataset(
        unprocessed_ds=train_orig, split="train", phone_vec=False, recompute_stats=False
    )

    model = FastSpeech2.load_from_checkpoint(config["model"].get("path")).to("cuda:0")
    synthesiser = Synthesiser(config["model"].getint("sampling_rate"), device="cuda:1")

    total_len = 0
    max_len = size * 60 * 60

    augmentation = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        ]
    )

    if speaker_diversity is not None:
        speakers = list(model.train_ds.speaker_map.values())
        if speaker_diversity == "decrease":
            single_speaker = speakers[np.random.randint(0, len(speakers))]
            print(f"only using speaker {single_speaker}")

    speaker_set = set()
    if num_speakers is not None:
        num_speakers = int(num_speakers)

    dvecs, speakers = train_orig_p.get_speaker_dvectors()

    speaker_dict = {k: v for v, k in zip(dvecs, speakers)}

    if num_speakers is not None:
        chosen_speakers = random.sample(list(speaker_dict.keys()), num_speakers)
        chosen_speakers = random.choices(chosen_speakers, k=len(speaker_dict))
        chosen_speakers = [speaker_dict[c] for c in chosen_speakers]
        for key, speaker in zip(speaker_dict.keys(), chosen_speakers):
            speaker_dict[key] = speaker

    p = Pool()

    if copy or resynthesise:
        loader = DataLoader(
            train_orig_p,
            batch_size=6,
            collate_fn=train_orig_p.collate_fn,
        )
    else:
        loader = model.train_dataloader()

    with tqdm(total=round(max_len / 60 / 60, 3)) as pbar:
        for batch in loader:
            if copy:
                batches = [batch["id"][i] for i in range(len(batch["speaker"]))]
                lens = p.map(copy_reference, batches)
                for l in lens:
                    total_len += l
                pbar.update(round(total_len / 60 / 60, 3) - pbar.n)

                if total_len >= max_len:
                    break
                continue

            speaker_strs = [
                batch["id"][i].split("_")[0] for i in range(len(batch["id"]))
            ]
            speaker_strs = [
                x if x in speaker_dict else random.choice(list(speaker_dict.keys()))
                for x in speaker_strs
            ]
            batch["speaker"] = torch.tensor([speaker_dict[x] for x in speaker_strs])
            preds, src_mask, tgt_mask = model(
                {
                    "phones": batch["phones"],
                    "speaker": batch["speaker"],
                }
            )
            old_tgt_mask = tgt_mask
            i = 0
            pitch, energy, duration = (
                preds["pitch"],
                preds["energy"],
                preds["duration_rounded"],
            )

            if duration_augment is not None:
                for p_i, phone_seq in enumerate(batch["phones"]):
                    phones = [model.train_ds.id2phone[int(p)] for p in phone_seq]
                    if duration_augment == "sample":
                        raise NotImplementedError()

            if duration_diversity is not None:
                duration = duration.float().cpu()
                if duration_diversity == "increase":
                    duration *= np.random.uniform(0.5, 1.5, size=duration.shape)
                    duration = torch.round(duration).int()
                if duration_diversity == "decrease":
                    duration[~src_mask] = segment1d(duration[~src_mask])
                    duration = duration.int()

            if pitch_diversity is not None:
                pitch = pitch.float().detach().cpu()
                if pitch_diversity == "increase":
                    pitch *= np.random.uniform(0.5, 1.5, size=pitch.shape)
                if pitch_diversity == "decrease":
                    pitch[~src_mask] = segment1d(pitch[~src_mask])

            if energy_diversity is not None:
                energy = energy.float().detach().cpu()
                if energy_diversity == "increase":
                    energy *= np.random.uniform(0.5, 1.5)
                if energy_diversity == "decrease":
                    energy[~src_mask] = segment1d(energy[~src_mask])

            if lexical_diversity is not None:
                raise NotImplementedError()

            if speaker_diversity == "increase":
                batch["speaker"] = torch.Tensor(
                    [
                        speakers[np.random.randint(0, len(speakers))]
                        for _ in batch["speaker"]
                    ]
                ).int()
            if speaker_diversity == "decrease":
                batch["speaker"] = torch.Tensor(
                    [single_speaker for _ in batch["speaker"]]
                ).int()

            if oracle == "duration" or oracle == "all":
                duration = batch["duration"]
            if oracle == "energy" or oracle == "all":
                energy = batch["energy"]
            if oracle == "pitch" or oracle == "all":
                pitch = batch["pitch"]

            if any(
                [
                    x is not None
                    for x in [
                        oracle,
                        energy_diversity,
                        pitch_diversity,
                        duration_diversity,
                        duration_augment,
                    ]
                ]
            ):
                preds, src_mask, tgt_mask = model(
                    {
                        "phones": batch["phones"],
                        "speaker": batch["speaker"],
                        "pitch": pitch,
                        "energy": energy,
                        "duration": duration,
                    }
                )

            if resynthesise:
                preds, src_mask, tgt_mask = model(batch)

            for i in range(len(batch["speaker"])):
                if not resynthesise:
                    mel = preds[0][i][~tgt_mask[i]]
                else:
                    mel = batch["mel"][i][~tgt_mask[i]]

                audio = synthesiser(mel)

                if snr_threshold is not None and wada_snr(audio[0]) < snr_threshold:
                    continue
                total_len += audio.shape[1] / float(
                    config["model"].getint("sampling_rate")
                )

                speaker_str = speaker_strs[i]
                Path(os.path.join(destination, speaker_str)).mkdir(exist_ok=True)

                if augment:
                    audio = augmentation(
                        int16_samples_to_float32(audio),
                        sample_rate=config["model"].getint("sampling_rate"),
                    )

                torchaudio.save(
                    os.path.join(destination, speaker_str, f'{batch["id"][i]}'),
                    torch.tensor(audio),
                    config["model"].getint("sampling_rate"),
                )
                with open(
                    os.path.join(destination, speaker_str, f'{batch["id"][i]}').replace(
                        ".wav", ".lab"
                    ),
                    "w",
                ) as lab:
                    lab.write(batch["text"][i])

            pbar.update(round(total_len / 60 / 60, 3) - pbar.n)

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
