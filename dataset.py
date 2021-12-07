import configparser
import os
import json

config = configparser.ConfigParser()
config.read("config.ini")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pandarallel import pandarallel
import plotly.express as px
import plotly.graph_objects as go
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch

from ipa_utils import get_phone_vecs

pandarallel.initialize(progress_bar=True)
tqdm.pandas()

# TODO: convert to pl DataModule


class ProcessedDataset(Dataset):
    def __init__(self, path, split, phone_map=None, phone_vec=False, phone2id=None):
        super().__init__()
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
        with open(os.path.join(self.path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
            self.speaker_n = len(self.speaker_map)
        with open(os.path.join(self.path, "stats.json")) as f:
            self.stats = json.load(f)

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
                    self.path,
                    type,
                    "{}-{}-{}.npy".format(speaker, type, basename),
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
        phones = self.data.iloc[idx]["phones"]

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "phones": phones,
            "text": text,
            "mel": self.get_values("mel", speaker, basename).float(),
            "pitch": self.get_values("pitch", speaker, basename).float(),
            "energy": self.get_values("energy", speaker, basename).float(),
            "duration": self.get_values("duration", speaker, basename).long(),
        }

        return sample

    @staticmethod
    def expand(values, durations):
        out = []
        for value, d in zip(values, durations):
            out += [value] * max(0, int(d))
        return np.array(out)

    def plot(self, sample):
        mel = sample["mel"]

        pitch = ProcessedDataset.expand(sample["pitch"], sample["duration"])
        pitch_min, pitch_max = self.stats["pitch"][:2]
        pitch = (pitch - pitch_min) / (pitch_max - pitch_min) * mel.shape[1]

        energy = ProcessedDataset.expand(sample["energy"], sample["duration"])
        energy_min, energy_max = self.stats["energy"][:2]
        energy = (energy - energy_min) / (energy_max - energy_min) * mel.shape[1]

        fig = px.imshow(mel.T, color_continuous_scale="gray", origin="lower")
        fig.add_trace(go.Scatter(x=np.arange(mel.shape[0]), y=pitch, name="pitch"))
        fig.add_trace(go.Scatter(x=np.arange(mel.shape[0]), y=energy, name="energy"))
        fig.update_yaxes(fixedrange=True)
        return fig

    def collate_fn(self, data):
        # list-of-dict -> dict-of-lists (see https://stackoverflow.com/a/33046935)
        data = {k: [dic[k] for dic in data] for k in data[0]}
        for key in ["phones", "mel", "pitch", "energy", "duration"]:
            data[key] = pad_sequence(data[key], batch_first=True, padding_value=0)
        data["speaker"] = torch.tensor(data["speaker"]).long()
        return data
