import configparser
import os
import json
from pathlib import Path

config = configparser.ConfigParser()
config.read("config.ini")

import torchdatasets as td
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pandarallel import pandarallel
import wandb
import plotly.express as px
import plotly.graph_objects as go

from ipa_utils import get_phone_vecs

pandarallel.initialize(progress_bar=True)
tqdm.pandas()

class ProcessedDataset(td.Dataset):
    def __init__(self, path, split, phone_map=None, phone_vec=False):
        super().__init__()
        self.path = path
        self.data = pd.read_csv(
            os.path.join(path,split)+".txt",
            sep="|",
            names=["basename", "speaker", "phones", "text"],
            dtype={"speaker": str},
        )
        self.data["phones"] = self.data["phones"].apply(lambda x: x.replace("{", "").replace("}", "").strip().split())
        self.data["text"] = self.data["text"].apply(lambda x: x.strip())
        self.phone_map = phone_map
        if phone_map is not None:
            self.data["phones"] = self.data["phones"].apply(self.apply_phone_map)
        if phone_vec:
            print("vectorizing phones")
            self.data["phones"] = self.data["phones"].parallel_apply(get_phone_vecs)
        with open(os.path.join(self.path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.path, "stats.json")) as f:
            self.stats = json.load(f)
            

    def apply_phone_map(self, phones):
        return [self.phone_map[p] if p in self.phone_map else p for p in phones]

    def get_values(self, type, speaker, basename):
        return np.load(os.path.join(
            self.path,
            type,
            "{}-{}-{}.npy".format(speaker, type, basename),
        ))

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
            "mel": self.get_values("mel", speaker, basename),
            "pitch": self.get_values("pitch", speaker, basename),
            "energy": self.get_values("energy", speaker, basename),
            "duration": self.get_values("duration", speaker, basename),
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

        fig = px.imshow(mel.T, color_continuous_scale='gray', origin="lower")
        fig.add_trace(
            go.Scatter(x=np.arange(mel.shape[0]), y=pitch, name="pitch")
        )
        fig.add_trace(
            go.Scatter(x=np.arange(mel.shape[0]), y=energy, name="energy")
        )
        fig.update_yaxes(fixedrange=True)
        return fig

if __name__ == "__main__":
    ds = ProcessedDataset("./data/GlobalPhoneGerman", "train", phone_vec=True).cache(td.cachers.Pickle(Path(".cache")))
    print(ds[100])
    # fig = ds.plot(ds[0])
    # wandb.init(project="test")
    # wandb.log({"test": fig})
