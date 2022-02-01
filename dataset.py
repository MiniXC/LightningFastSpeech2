import configparser
import os

import json
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio
import hifigan
from glob import glob

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pandarallel import pandarallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch
import tgt
import pyworld as pw
import librosa

from ipa_utils import get_phone_vecs
from audio_utils import (
    dynamic_range_compression,
    remove_outliers,
    remove_outliers_new,
    smooth,
    get_alignment
)

config = configparser.ConfigParser()
config.read("config.ini")

pandarallel.initialize(progress_bar=True)
tqdm.pandas()


# TODO: convert to pl DataModules

class UnprocessedDataset(Dataset):
    def __init__(self, directory, extract_speaker=lambda x: x.split("/")[-1]):
        super().__init__()
        df_dict = {
            "phones": [],
            "durations": [],
            "start": [],
            "end": [],
            "audio": [],
            "speaker": [],
            "transcription": [],
        }

        self.sampling_rate = config["dataset"].getint("sampling_rate")
        self.n_fft = config["dataset"].getint("n_fft")
        self.win_length = config["dataset"].getint("win_length")
        self.hop_length = config["dataset"].getint("hop_length")
        self.n_mels = config["dataset"].getint("n_mels")
        self.f_max = config["dataset"].getint("f_max")
        self.dio_speed = config["dataset"].getint("dio_speed")
        self.pitch_smooth = config["dataset"].getint("pitch_smooth")
        self.new_outlier_method = config["dataset"].getboolean("new_outlier_method")

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_max=self.f_max,
        ).to("cuda")

        tg_missing = 0

        for audio_file in tqdm(
            sorted(glob(os.path.join(directory, "**", "*.wav")))
        ):
            try:
                textgrid = tgt.io.read_textgrid(
                    audio_file.replace(".wav", ".TextGrid")
                )
            except FileNotFoundError:
                tg_missing += 1
                continue
            phones, durations, start, end = get_alignment(
                textgrid.get_tier_by_name("phones"),
                self.sampling_rate,
                self.hop_length,
            )
            if start >= end:
                continue
            df_dict["phones"].append(phones)
            df_dict["durations"].append(durations)
            df_dict["start"].append(start)
            df_dict["end"].append(end)
            df_dict["audio"].append(audio_file)
            df_dict["speaker"].append(extract_speaker(audio_file))
            transcription = open(
                audio_file.replace(".wav", ".lab"), "r"
            ).read()
            df_dict["transcription"].append(transcription)
        print(f"{tg_missing} textgrids not found")
        self.data = pd.DataFrame(df_dict)
        del df_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, sampling_rate = torchaudio.load(row["audio"])
        assert sampling_rate == self.sampling_rate
        start = int(self.sampling_rate * row["start"])
        end = int(self.sampling_rate * row["end"])
        audio = torch.Tensor([audio[0][start:end]])
        mel = dynamic_range_compression(
            self.mel_spectrogram(audio.to("cuda"))
        )
        energy = librosa.feature.rms(
            audio,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )
        pitch, t = pw.dio(
                audio.astype(np.float64),
                self.sampling_rate,
                frame_period=self.hop_length / self.sampling_rate * 1000,
                speed=self.dio_speed,
            )
        pitch = pw.stonemask(
            audio.astype(np.float64), pitch, t, self.sampling_rate
        )
        if self.new_outlier_method:
            pitch = remove_outliers_new(pitch)
        else:
            pitch = remove_outliers(pitch)
        if self.pitch_smooth > 1:
            pitch = smooth(pitch, self.pitch_smooth)
        plt.imshow(mel.T, origin="lower")
        sns.lineplot(
            x=list(range(len(mel))) + list(range(len(mel))),
            y=list(pitch) + list(energy),
            hue=["Pitch"] * len(mel) + ["Energy"] * len(mel),
            palette="inferno",
        )
        plt.ylim(0)
        plt.yticks(range(0, 81, 10))
        plt.show()
        return {
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
        }


class ProcessedDataset(Dataset):
    def __init__(
        self, path, split, phone_map=None, phone_vec=False, phone2id=None
    ):
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
            self.data["phones"] = self.data["phones"].apply(
                self.apply_phone_map
            )
        if phone_vec:
            print("vectorizing phones")
            self.data["phones"] = self.data["phones"].parallel_apply(
                get_phone_vecs
            )
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

        if split == "val":
            with open("hifigan/config.json", "r") as f:
                config = json.load(f)
            config["sampling_rate"] = 22050
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator(config)
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(torch.device("cuda"))
            self.vocoder = vocoder

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
        return [
            self.phone_map[p] if p in self.phone_map else p for p in phones
        ]

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
        energy = (
            (energy - energy_min) / (energy_max - energy_min) * mel.shape[1]
        )

        plt.imshow(mel.T, origin="lower")
        sns.lineplot(
            x=list(range(len(mel))) + list(range(len(mel))),
            y=list(pitch) + list(energy),
            hue=["Pitch"] * len(mel) + ["Energy"] * len(mel),
            palette="inferno",
        )
        plt.ylim(0)
        plt.yticks(range(0, 81, 10))
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.clf()
        buf.seek(0)
        return Image.open(buf)

    def synthesise(self, mel):
        mel = torch.unsqueeze(mel.T.cuda(), 0)
        return (self.vocoder(mel).squeeze(1).cpu().numpy() * 32768.0).astype(
            "int16"
        )

    def collate_fn(self, data):
        # list-of-dict -> dict-of-lists
        # (see https://stackoverflow.com/a/33046935)
        data = {k: [dic[k] for dic in data] for k in data[0]}
        for key in ["phones", "mel", "pitch", "energy", "duration"]:
            data[key] = pad_sequence(
                data[key], batch_first=True, padding_value=0
            )
        data["speaker"] = torch.tensor(data["speaker"]).long()
        return data


if __name__ == "__main__":
    ds = UnprocessedDataset("/mnt/D4C0-9435/mls_german/processed")
    ds[0]