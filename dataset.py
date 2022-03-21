from audioop import mul
import configparser
import os

import json
from pathlib import Path
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
import scipy
import seaborn as sns
import torchaudio
from glob import glob
from phones.convert import Converter
from phones import PhoneCollection
from scipy import signal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pandarallel import pandarallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map
import torchaudio.transforms as T
import torchaudio.functional as F
import torch
import tgt
import pyworld as pw
from torchaudio import transforms
import multiprocessing
from random import Random
from copy import deepcopy

from ipa_utils import get_phone_vecs
from audio_utils import (
    dynamic_range_compression,
    remove_outliers,
    remove_outliers_new,
    smooth,
    get_alignment,
)

from librosa.filters import mel as librosa_mel

matplotlib.use('AGG', force=True)

config = configparser.ConfigParser()
config.read("config.ini")

pandarallel.initialize(progress_bar=True)
tqdm.pandas()

# TODO: convert to pl DataModules
class UnprocessedDataset(Dataset):
    def __init__(self, audio_directory, alignment_directory=None, max_entries=None):
        super().__init__()

        self.dir = audio_directory

        if alignment_directory is None:
            alignment_directory=audio_directory

        self.sampling_rate = config["dataset"].getint("sampling_rate")
        self.n_fft = config["dataset"].getint("n_fft")
        self.win_length = config["dataset"].getint("win_length")
        self.hop_length = config["dataset"].getint("hop_length")
        self.n_mels = config["dataset"].getint("n_mels")
        self.f_min = config["dataset"].getint("f_min")
        self.f_max = config["dataset"].getint("f_max")
        self.dio_speed = config["dataset"].getint("dio_speed")
        self.pitch_smooth = config["dataset"].getint("pitch_smooth")
        self.new_outlier_method = config["dataset"].getboolean("new_outlier_method")

        self.target_lang = config["dataset"].get("target_lang")
        if self.target_lang == "None":
            self.target_lang = None
        self.remove_stress = config["dataset"].getboolean("remove_stress")
        self.source_phoneset = config["dataset"].get("source_phoneset")

        self.converter = Converter()
        self.pc = PhoneCollection()

        self.phone_cache = {}

        self.grid_files = {
            Path(file).name.replace(".TextGrid", ".wav"): file
            for file
            in glob(os.path.join(alignment_directory, "**/*.TextGrid"), recursive=True)
        }

        self.grid_missing = 0

        if max_entries is not None:
            entry_list = list(glob(os.path.join(audio_directory, "**", "*.wav")))[:max_entries]
        else:
            entry_list = list(glob(os.path.join(audio_directory, "**", "*.wav")))

        Random(42).shuffle(entry_list)

        entries = [
            entry
            for entry in process_map(
                self.create_entry,
                entry_list,
                chunksize=100,
                max_workers=multiprocessing.cpu_count(),
                desc="collecting textgrid and audio files"
            )
            if entry is not None
        ]

        print(f"{self.grid_missing} textgrids not found")
        self.data = pd.DataFrame(entries, columns=["phones", "duration", "start", "end", "audio", "speaker", "text", "basename"])
        del entries

        self.mel_spectrogram = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=0,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True)
        self.mel_basis = librosa_mel(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()

    def create_entry(self, audio_file):
        extract_speaker = lambda x: x.split("/")[-2]

        try:
            audio_name = Path(audio_file).name
            grid_file = self.grid_files[audio_name]
            textgrid = tgt.io.read_textgrid(grid_file)
        except KeyError:
            self.grid_missing += 1
            return None
        phones, durations, start, end = get_alignment(
            textgrid.get_tier_by_name("phones"), self.sampling_rate, self.hop_length,
        )
        for i, phone in enumerate(phones):
            add_stress = False
            if self.remove_stress:
                r_phone = phone.replace('0', '').replace('1', '')
            else:
                # TODO: this does not work properly yet, we'd need the syllable boundary
                r_phone = phone.replace('0', '').replace('1', 'ˈ')
                if 'ˈ' in r_phone:
                    add_stress = True
            if len(r_phone) > 0:
                phone = r_phone
            if phone not in ["spn", "sil"]:
                o_phone = phone
                if o_phone not in self.phone_cache:
                    phone = self.converter(phone, self.source_phoneset, lang=self.target_lang)[0]
                    self.phone_cache[o_phone] = phone
                phone = self.phone_cache[o_phone]
                if add_stress:
                    phone = 'ˈ' + phone
            else:
                phone = "[SILENCE]"
            phones[i] = phone
        if start >= end:
            return None
        try:
            with open(audio_file.replace(".wav", ".lab"), "r") as f:
                transcription = f.read()
        except UnicodeDecodeError:
            print("failed to read")
            return None
        return (
            phones,
            durations,
            start,
            end,
            audio_file,
            extract_speaker(audio_file),
            transcription,
            audio_name
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, sampling_rate = torchaudio.load(row["audio"])
        if sampling_rate != self.sampling_rate:
            transform = transforms.Resample(sampling_rate, self.sampling_rate)
            audio = transform(audio)
        start = int(self.sampling_rate * row["start"])
        end = int(self.sampling_rate * row["end"])

        audio = audio[0][start:end]
        audio = torch.clip(audio, -1, 1)
        
        mel = self.mel_spectrogram(audio.unsqueeze(0))
        mel = torch.sqrt(mel[0])
        energy = torch.norm(mel, dim=0).cpu()
        
        mel = torch.matmul(self.mel_basis, mel)
        mel = dynamic_range_compression(mel).cpu()

        pitch, t = pw.dio(
            audio.cpu().numpy().astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
            speed=self.dio_speed,
        )
        pitch = pw.stonemask(audio.cpu().numpy().astype(np.float64), pitch, t, self.sampling_rate)

        if self.new_outlier_method:
            old_pitch = pitch
            pitch = remove_outliers_new(pitch)
            energy = remove_outliers_new(energy)
            if pitch is None:
                # TODO: maybe drop instead?
                pitch = old_pitch
        else:
            pitch = remove_outliers(pitch)
            energy = remove_outliers(energy)
        if self.pitch_smooth > 1:
            pitch = smooth(pitch, self.pitch_smooth)

        duration = row["duration"]
        
        # TODO: investigate why this is necessary
        pitch = torch.tensor(pitch.astype(np.float32))[:sum(duration)]
        energy = energy[:sum(duration)]

        return {
            "mel": np.array(mel.T)[:sum(duration)],
            "pitch": np.array(pitch),
            "energy": np.array(energy),
            "duration": np.array(duration),
        }

    def plot(self, sample, show=False):
        mel = sample["mel"]
        pitch = sample["pitch"]
        energy = sample["energy"]
        plt.imshow(mel, origin="lower")
        sns.lineplot(
            x=list(range(len(pitch))) + list(range(len(energy))),
            y=list((pitch-pitch.min())*70) + list((energy-energy.min())+70),
            hue=["Pitch"] * len(pitch) + ["Energy"] * len(energy),
            palette="inferno",
        )
        plt.ylim(0,80)
        plt.yticks(range(0, 81, 10))
        if show:
            plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.clf()
        buf.seek(0)
        return Image.open(buf)


class ProcessedDataset(Dataset):
    def __init__(self, path=None, split=None, phone_map=None, phone_vec=False, phone2id=None, unprocessed_ds=None, stats=None, recompute_stats=False):
        super().__init__()
        self.stats = None
        if unprocessed_ds is None:
            self.path = path
            self.data = pd.read_csv(
                os.path.join(path, split) + ".txt",
                sep="|",
                names=["basename", "speaker", "phones", "text"],
                dtype={"speaker": str},
            )
            self.data["phones"] = self.data["phones"].apply(
                lambda x: x.replace("{", "").replace("}", "").strip().split()
            )
            
            with open(os.path.join(self.path, "speakers.json")) as f:
                self.speaker_map = json.load(f)
                self.speaker_n = len(self.speaker_map)
            with open(os.path.join(self.path, "stats.json")) as f:
                self.stats = json.load(f)
        else:
            self.ds = unprocessed_ds
            self.data = unprocessed_ds.data
            speakers = self.data["speaker"].unique()
            self.speaker_map = {s: i for s, i in zip(speakers, range(len(speakers)))}
            self.speaker_n = len(self.speaker_map)

            if recompute_stats or (not Path(self.ds.dir, 'stats.json').exists() and stats is None):
                p_stats = process_map(
                    self._get_stats,
                    range(len(self.ds)),
                    chunksize=100,
                    max_workers=multiprocessing.cpu_count(),
                    desc="computing stats (this is only done once)"
                )

                stat_json = {
                    'pitch_min': np.min([s['pitch_min'] for s in p_stats]),
                    'pitch_max': np.max([s['pitch_max'] for s in p_stats]),
                    'pitch_mean': np.mean([s['pitch_mean'] for s in p_stats]),
                    'pitch_std': np.mean([s['pitch_std'] for s in p_stats]),
                    'energy_min': np.min([s['energy_min'] for s in p_stats]),
                    'energy_max': np.max([s['energy_max'] for s in p_stats]),
                    'energy_mean': np.mean([s['energy_mean'] for s in p_stats]),
                    'energy_std': np.mean([s['energy_std'] for s in p_stats]),
                }

                with open(os.path.join(self.ds.dir, 'stats.json'), 'w') as outfile:
                    json.dump(stat_json, outfile)
            
            if stats is None:
                with open(os.path.join(self.ds.dir, 'stats.json')) as f:
                    self.stats = json.load(f)
            else:
                self.stats = stats

        self.data["text"] = self.data["text"].apply(lambda x: x.strip())
        self.phone_map = phone_map
        if phone_map is not None:
            self.data["phones"] = self.data["phones"].apply(self.apply_phone_map)
        if phone_vec:
            print("vectorizing phones")
            self.data["phones"] = self.data["phones"].parallel_apply(get_phone_vecs)
        else:
            if phone2id is not None:
                self.phone2id = phone2id
            else:
                self.create_phone2id()
            self.vocab_n = len(self.phone2id)
            self.data["phones"] = self.data["phones"].apply(
                lambda x: torch.tensor([self.phone2id[p] for p in x]).long()
            )

        self.id2phone = {v:k for k,v in self.phone2id.items()}

    def _get_stats(self, idx):
        x = self.ds[idx]
        return {
            'pitch_min': np.min(x['pitch']).astype(float),
            'pitch_max': np.max(x['pitch']).astype(float),
            'pitch_mean': np.mean(x['pitch']).astype(float),
            'pitch_std':np.std(x['pitch']).astype(float),
            'energy_min':np.min(x['energy']).astype(float),
            'energy_max':np.max(x['energy']).astype(float),
            'energy_mean':np.mean(x['energy']).astype(float),
            'energy_std':np.std(x['energy']).astype(float),
        }

    def create_phone2id(self):
        unique_phones = set()
        for phone_list in self.data["phones"]:
            unique_phones.update(phone_list)
        unique_phones = list(sorted(unique_phones))
        self.phone2id = {
            p: i + 1 for p, i in zip(unique_phones, range(len(unique_phones)))
        }
        self.phone2id["[PAD]"] = 0

    def apply_phone_map(self, phones):
        return [self.phone_map[p] if p in self.phone_map else p for p in phones]

    def get_values(self, type, speaker, basename):
        return torch.from_numpy(
            np.load(
                os.path.join(
                    self.path, type, "{}-{}-{}.npy".format(speaker, type, basename),
                )
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        basename = self.data.iloc[idx]["basename"]
        speaker = self.data.iloc[idx]["speaker"]
        speaker_id = self.speaker_map[speaker]
        text = self.data.iloc[idx]["text"]
        phones = deepcopy(self.data.iloc[idx]["phones"])

        entry = self.ds[idx]

        if config["dataset"].get("variance_level") == "phoneme":
            for feature in ["pitch" , "energy"]:
                pos = 0
                for i, d in enumerate(entry["duration"]):
                    if d > 0:
                        entry[feature][i] = np.mean(entry[feature][pos : pos + d])
                    else:
                        entry[feature][i] = 0
                    pos += d
                entry[feature] = entry[feature][:len(entry["duration"])]

        if self.stats is not None:
            entry["pitch"] = (entry["pitch"] - self.stats['pitch_mean']) / self.stats['pitch_std']
            entry["energy"] = (entry["energy"] - self.stats['energy_mean']) / self.stats['energy_std']

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "phones": phones,
            "text": text,
            "mel": entry["mel"],
            "pitch": entry["pitch"],
            "energy": entry["energy"],
            "duration": entry["duration"],
        }

        return sample

    @staticmethod
    def expand(values, durations):
        out = []
        for value, d in zip(values, durations):
            out += [value] * max(0, int(d))
        return np.array(out)

    def plot(self, sample, show=False):
        mel = sample["mel"]

        if config["dataset"].get("variance_level") == "phoneme":
            pitch = ProcessedDataset.expand(sample["pitch"], sample["duration"])[:len(mel)]
            energy = ProcessedDataset.expand(sample["energy"], sample["duration"])[:len(mel)]
        elif config["dataset"].get("variance_level") == "frame":
            pitch = sample["pitch"][:len(mel)]
            energy = sample["energy"][:len(mel)]

        pitch_min, pitch_max = self.stats["pitch_min"], self.stats["pitch_max"]
        pitch = (pitch - pitch_min) / (pitch_max - pitch_min) * mel.shape[1]

        energy_min, energy_max = self.stats["energy_min"], self.stats["energy_max"]
        energy = (energy - energy_min) / (energy_max - energy_min) * mel.shape[1]

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.imshow(mel.T, origin="lower")
        sns.lineplot(
            x=list(range(len(pitch))) + list(range(len(energy))),
            y=list((pitch-pitch.min())*70) + list((energy-energy.min())*70),
            hue=["Pitch"] * len(pitch) + ["Energy"] * len(energy),
            palette="inferno",
            ax=ax,
        )
        plt.ylim(0, 80)
        plt.yticks(range(0, 81, 10))

        plt.ylim(0, 80)
        plt.yticks(range(0, 81, 10))
        last = 0

        for phone, duration in zip(sample["phones"], sample["duration"]):
            x = last
            ax.axline((int(x), 0), (int(x), 80))
            phone = self.id2phone[int(phone)]
            # print(int(x), phone)
            ax.text(int(x), 40, phone)
            last += duration
        if show:
            plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.clf()
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def collate_fn(self, data):
        # list-of-dict -> dict-of-lists
        # (see https://stackoverflow.com/a/33046935)
        data = {k: [dic[k] for dic in data] for k in data[0]}
        for key in ["phones", "mel", "pitch", "energy", "duration"]:
            for t in data[key]:
                if isinstance(t, list):
                    print(t)
                # print(key, t.shape)
            if torch.is_tensor(data[key][0]):
                data[key] = pad_sequence(data[key], batch_first=True, padding_value=0)
            else:
                data[key] = pad_sequence([torch.tensor(x) for x in data[key]], batch_first=True, padding_value=0)
        data["speaker"] = torch.tensor(data["speaker"]).long()
        return data


if __name__ == "__main__":

    train_path = config["train"].get("train_path")
    valid_path = config["train"].get("valid_path")
    train_ud = UnprocessedDataset(train_path, max_entries=1000)
    valid_ud = UnprocessedDataset(valid_path)
    train_ds = ProcessedDataset(
        unprocessed_ds=train_ud,
        split="train",
        phone_vec=False,
        recompute_stats=True,
    )
    valid_ds = ProcessedDataset(
        unprocessed_ds=valid_ud,
        split="val",
        phone_vec=False,
        phone2id=train_ds.phone2id,
        stats=train_ds.stats
    )
    valid_ds.plot(valid_ds[0], show=True)
