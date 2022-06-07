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
from tqdm.contrib.concurrent import process_map, thread_map
import torchaudio.transforms as T
import torchvision.transforms as VT
import torchaudio.functional as F
import torch
import tgt
import pyworld as pw
from torchaudio import transforms
import multiprocessing
from random import Random
from copy import deepcopy
from wav2mel.wav2mel import Wav2Mel
from time import time
from pathlib import Path
from snr import WADA

from multiprocessing import Pool, get_context

from ipa_utils import get_phone_vecs
from audio_utils import (
    dynamic_range_compression,
    remove_outliers,
    smooth,
    get_alignment,
)

from librosa.filters import mel as librosa_mel

config = configparser.ConfigParser()
config.read("config.ini")

pandarallel.initialize(progress_bar=True)
tqdm.pandas()


class Timer:
    def __init__(self, name):
        self.name = name
        self.duration = 0

    def __enter__(self):
        self.start = time()

    def __exit__(self, type, value, traceback):
        self.duration = time() - self.start
        print(f"{self.duration} - {self.name}")


# TODO: convert to pl DataModules
class UnprocessedDataset(Dataset):
    def __init__(
        self,
        audio_directory,
        alignment_directory=None,
        max_entries=None,
        dvector=False,
        dvector_mean=True,
        treshold=0.5,
        treshold_max=32,
        no_alignments=False,
        augment_duration=0.1,
        use_snr=False,
        conditioned=False,
    ):
        super().__init__()

        self.augment_duration = augment_duration
        self.use_snr = use_snr

        self.dvector_mean = dvector_mean

        self.dir = audio_directory

        self.treshold = treshold
        self.treshold_max = treshold_max

        if alignment_directory is None:
            alignment_directory = audio_directory

        self.sampling_rate = config["dataset"].getint("sampling_rate")
        self.n_fft = config["dataset"].getint("n_fft")
        self.win_length = config["dataset"].getint("win_length")
        self.hop_length = config["dataset"].getint("hop_length")
        self.n_mels = config["dataset"].getint("n_mels")
        self.f_min = config["dataset"].getint("f_min")
        self.f_max = config["dataset"].getint("f_max")
        self.dio_speed = config["dataset"].getint("dio_speed")
        self.pitch_smooth = config["dataset"].getint("pitch_smooth")
        self.remove_outliers = config["dataset"].getboolean("remove_outliers")
        self.pitch_type = config["dataset"].get("pitch_type")

        self.target_lang = config["dataset"].get("target_lang")
        if self.target_lang == "None":
            self.target_lang = None
        self.remove_stress = config["dataset"].getboolean("remove_stress")
        self.source_phoneset = config["dataset"].get("source_phoneset")

        self.converter = Converter()
        self.pc = PhoneCollection()

        self.has_dvector = dvector

        self.phone_cache = {}

        self.no_alignments = no_alignments
        if not no_alignments:
            self.grid_files = {
                Path(file).name.replace(".TextGrid", ".wav"): file
                for file in glob(
                    os.path.join(alignment_directory, "**/*.TextGrid"), recursive=True
                )
            }

            self.grid_missing = 0

        if max_entries is not None:
            entry_list = list(glob(os.path.join(audio_directory, "**", "*.wav")))[
                :max_entries
            ]
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
                desc="collecting textgrid and audio files",
            )
            if entry is not None
        ]

        if not no_alignments:
            print(f"{self.grid_missing} textgrids not found")
        self.data = pd.DataFrame(
            entries,
            columns=[
                "phones",
                "duration",
                "start",
                "end",
                "audio",
                "speaker",
                "text",
                "basename",
            ],
        )
        del entries

        if self.has_dvector:
            dvecs = []
            for i, row in self.data.iterrows():
                if not self.dvector_mean:
                    dvecs.append(row["audio"].replace(".wav", ".npy"))
                else:
                    dvecs.append(Path(row["audio"]).parent / "speaker.npy")
            self.data["dvector"] = dvecs

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
            onesided=True,
        )
        self.mel_basis = librosa_mel(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()

        if self.has_dvector:
            self.create_dvectors()

        self.wada = WADA()

        self.conditioned = conditioned

    def create_dvectors(self):
        wav2mel = Wav2Mel()
        dvector_gen = torch.jit.load("dvector/dvector.pt").eval()

        for idx, row in tqdm(
            self.data.iterrows(), desc="creating dvectors", total=len(self.data)
        ):
            dvec_path = Path(row["audio"].replace(".wav", ".npy"))
            if not dvec_path.exists():
                audio, sampling_rate = torchaudio.load(row["audio"])
                start = int(self.sampling_rate * row["start"])
                end = int(self.sampling_rate * (row["start"] + 1))
                audio = audio[0][start:end]
                audio = audio / torch.max(torch.abs(audio))  # might not be necessary

                if 16_000 != self.sampling_rate:
                    transform = transforms.Resample(sampling_rate, self.sampling_rate)
                    audio = transform(audio)
                dvector_mel = wav2mel(torch.unsqueeze(audio, 0).cpu(), 16_000)
                dvec_result = dvector_gen.embed_utterance(dvector_mel).detach()

                np.save(dvec_path, dvec_result.numpy())

        speaker_dvecs = []
        for speaker in tqdm(
            self.data["speaker"].unique(), desc="creating speaker dvectors"
        ):
            speaker_df = self.data[self.data["speaker"] == speaker]
            speaker_path = Path(speaker_df.iloc[0]["audio"]).parent / "speaker.npy"
            if not speaker_path.exists():
                dvecs = []
                for idx, row in speaker_df.iterrows():
                    dvecs.append(np.load(row["audio"].replace(".wav", ".npy")))
                dvec = np.mean(dvecs, axis=0)
                np.save(speaker_path, dvec)

    def create_entry(self, audio_file):
        extract_speaker = lambda x: x.split("/")[-2]
        audio_name = Path(audio_file).name

        if not self.no_alignments:
            try:
                grid_file = self.grid_files[audio_name]
                textgrid = tgt.io.read_textgrid(grid_file)
            except KeyError:
                self.grid_missing += 1
                return None
            ali = get_alignment(
                textgrid.get_tier_by_name("phones"),
                self.sampling_rate,
                self.hop_length,
            )
            if ali is None:
                return None

            phones, durations, start, end = ali

            if end - start < self.treshold or end - start > self.treshold_max:
                return None

            for i, phone in enumerate(phones):
                add_stress = False
                phone.replace("ˌ", "")
                if self.remove_stress:
                    r_phone = phone.replace("0", "").replace("1", "")
                else:
                    # TODO: this does not work properly yet, we'd need the syllable boundary
                    r_phone = phone.replace("0", "").replace("1", "ˈ")
                    if "ˈ" in r_phone:
                        add_stress = True
                if len(r_phone) > 0:
                    phone = r_phone
                if phone not in ["spn", "sil"]:
                    o_phone = phone
                    if o_phone not in self.phone_cache:
                        phone = self.converter(
                            phone, self.source_phoneset, lang=self.target_lang
                        )[0]
                        self.phone_cache[o_phone] = phone
                    phone = self.phone_cache[o_phone]
                    if add_stress:
                        phone = "ˈ" + phone
                else:
                    phone = "[SILENCE]"
                phones[i] = phone
            if start >= end:
                return None
        else:
            phones = None
            durations = None
            start = 0
            audio, sampling_rate = torchaudio.load(audio_file)
            end = len(audio[0]) / sampling_rate

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
            audio_name,
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
        audio = audio / torch.max(torch.abs(audio))

        if self.use_snr:
            snrs = np.clip(
                (self.wada.snr_windowed(audio, self.hop_length, self.win_length) - 40)
                / 60,
                -1.1,
                1.1,
            )

        mel = self.mel_spectrogram(audio.unsqueeze(0))
        mel = torch.sqrt(mel[0])
        energy = torch.norm(mel, dim=0).cpu()

        mel = torch.matmul(self.mel_basis, mel)
        mel = dynamic_range_compression(mel).cpu()

        if str(self.pitch_type) == "pyworld":
            pitch, t = pw.dio(
                audio.cpu().numpy().astype(np.float64),
                self.sampling_rate,
                frame_period=self.hop_length / self.sampling_rate * 1000,
                speed=self.dio_speed,
            )
            pitch = pw.stonemask(
                audio.cpu().numpy().astype(np.float64), pitch, t, self.sampling_rate
            )
        else:
            pitch = F.compute_kaldi_pitch(
                audio.unsqueeze(0),
                self.sampling_rate,
                self.win_length / self.sampling_rate * 1000,
                self.hop_length / self.sampling_rate * 1000,
            )[..., 0][0]

        if self.remove_outliers:
            pitch = remove_outliers(pitch)
            energy = remove_outliers(energy)
        if self.pitch_smooth > 1:
            pitch = smooth(pitch, self.pitch_smooth)

        duration = row["duration"]
        if self.augment_duration > 0:
            truth_vals = np.random.uniform(size=len(duration)) >= self.augment_duration
            rand_vals = np.random.normal(0, 1, size=len(duration)).round()
            rand_vals[truth_vals] = 0
            rand_vals_shifted = rand_vals[:-1] * -1
            rand_vals[1:] += rand_vals_shifted
            rand_vals = rand_vals.astype(int)
            rand_vals[(duration + rand_vals) < 0] = 0
            if rand_vals.sum() != 0:
                rand_vals[-1] -= rand_vals.sum()
                i = -1
                while True:
                    if rand_vals[i] < 0:
                        rand_vals[i - 1] += rand_vals[i]
                        rand_vals[i] = 0
                        i -= 1
                    else:
                        break
            duration += rand_vals
            duration[duration < 0] = 0

        if self.no_alignments:
            dur_sum = -1
        else:
            dur_sum = sum(duration)

        # TODO: investigate why this is necessary
        pitch = torch.tensor(pitch.astype(np.float32))[:dur_sum]
        energy = energy[:dur_sum]

        result = {
            "mel": np.array(mel.T)[:dur_sum],
            "pitch": np.array(pitch),
            "energy": np.array(energy),
            "duration": np.array(duration),
        }

        if self.use_snr:
            result["snr"] = snrs[:dur_sum].astype(np.float32)

        if self.has_dvector:
            result["dvector"] = row["dvector"]

        if self.conditioned:
            result["conditioned"] = {}
            for var in ["pitch", "energy", "duration"]:
                result["conditioned"][var] = np.mean(result[var])
            result["conditioned"]["snr"] = np.clip(
                (self.wada.snr(audio) - 40)
                / 60,
                -1.1,
                1.1,
            )

        return result


class ProcessedDataset(Dataset):
    def __init__(
        self,
        path=None,
        split=None,
        phone_map=None,
        phone_vec=False,
        phone2id=None,
        unprocessed_ds=None,
        stats=None,
        recompute_stats=False,
    ):
        super().__init__()
        self.stats = None
        if unprocessed_ds is None:
            self.ds = None
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

            if recompute_stats or (
                not Path(self.ds.dir, "stats.json").exists() and stats is None
            ):
                p_stats = process_map(
                    self._get_stats,
                    range(len(self.ds)),
                    chunksize=5,
                    max_workers=multiprocessing.cpu_count(),
                    desc="computing stats (this is only done once)",
                )
                # p_stats = []
                # for i in tqdm(range(len(self.ds)), desc="computing stats"):
                #     p_stats.append(self._get_stats(i))

                stat_json = {
                    "pitch_min": np.min([s["pitch_min"] for s in p_stats]),
                    "pitch_max": np.max([s["pitch_max"] for s in p_stats]),
                    "pitch_mean": np.mean([s["pitch_mean"] for s in p_stats]),
                    "pitch_std": np.mean([s["pitch_std"] for s in p_stats]),
                    "energy_min": np.min([s["energy_min"] for s in p_stats]),
                    "energy_max": np.max([s["energy_max"] for s in p_stats]),
                    "energy_mean": np.mean([s["energy_mean"] for s in p_stats]),
                    "energy_std": np.mean([s["energy_std"] for s in p_stats]),
                    "cond_duration_min": np.min([s["cond_duration"] for s in p_stats]),
                    "cond_duration_max": np.max([s["cond_duration"] for s in p_stats]),
                    "cond_pitch_min": np.min([s["cond_pitch"] for s in p_stats]),
                    "cond_pitch_max": np.max([s["cond_pitch"] for s in p_stats]),
                    "cond_energy_min": np.min([s["cond_energy"] for s in p_stats]),
                    "cond_energy_max": np.max([s["cond_energy"] for s in p_stats]),
                }

                with open(os.path.join(self.ds.dir, "stats.json"), "w") as outfile:
                    json.dump(stat_json, outfile)

            if stats is None:
                with open(os.path.join(self.ds.dir, "stats.json")) as f:
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
        elif not self.ds.no_alignments:
            if phone2id is not None:
                self.phone2id = phone2id
            else:
                self.create_phone2id()
            self.vocab_n = len(self.phone2id)
            self.data["phones"] = self.data["phones"].apply(
                lambda x: torch.tensor(
                    [self.phone2id[p.replace("ˌ", "")] for p in x]
                ).long()
            )

        if not self.ds.no_alignments:
            self.id2phone = {v: k for k, v in self.phone2id.items()}

    def _get_stats(self, idx):
        x = self.ds[idx]
        result = {
            "pitch_min": np.min(x["pitch"]).astype(float),
            "pitch_max": np.max(x["pitch"]).astype(float),
            "pitch_mean": np.mean(x["pitch"]).astype(float),
            "pitch_std": np.std(x["pitch"]).astype(float),
            "energy_min": np.min(x["energy"]).astype(float),
            "energy_max": np.max(x["energy"]).astype(float),
            "energy_mean": np.mean(x["energy"]).astype(float),
            "energy_std": np.std(x["energy"]).astype(float),
            "cond_duration": x["conditioned"]["duration"].astype(float),
            "cond_pitch": x["conditioned"]["pitch"].astype(float),
            "cond_energy": x["conditioned"]["energy"].astype(float),
        }
        if "snr" in x["conditioned"]:
            result["cond_snr"] = x["conditioned"]["snr"].astype(float)
        return result

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

        entry = self.ds[idx]

        if config["dataset"].get("variance_level") == "phoneme":
            features = ["pitch", "energy"]
            if self.ds.use_snr:
                features.append("snr")
            for feature in features:
                pos = 0
                for i, d in enumerate(entry["duration"]):
                    if d > 0:
                        if len(entry[feature][pos : pos + d]) > 0:
                            entry[feature][i] = np.mean(entry[feature][pos : pos + d])
                        else:
                            entry[feature][i] = -1.1
                    else:
                        entry[feature][i] = 0
                    pos += d
                entry[feature] = entry[feature][: len(entry["duration"])]

        if self.stats is not None:
            entry["pitch"] = (entry["pitch"] - self.stats["pitch_mean"]) / self.stats[
                "pitch_std"
            ]
            entry["energy"] = (
                entry["energy"] - self.stats["energy_mean"]
            ) / self.stats["energy_std"]

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

        if "dvector" in entry:
            sample["speaker"] = torch.from_numpy(np.load(entry["dvector"]))

        if "snr" in entry:
            sample["snr"] = entry["snr"]

        if self.ds.conditioned:
            if self.stats is not None:
                sample["cond_energy"] = entry["conditioned"]["pitch"]
                sample["cond_pitch"] = entry["conditioned"]["energy"]
                sample["cond_duration"] = entry["conditioned"]["duration"]
                if "snr" in entry["conditioned"]:
                    sample["cond_snr"] = entry["conditioned"]["snr"]

        sample_cp = deepcopy(sample)
        del entry
        del sample
        del phones

        return sample_cp

    def get_speaker_dvectors(self):
        speaker_dvecs = []
        speakers = []
        for speaker in tqdm(sorted(list(self.data["speaker"].unique()))):
            speaker_df = self.data[self.data["speaker"] == speaker]
            dvec_path = Path(speaker_df.iloc[0]["audio"]).parent / "speaker.npy"
            speaker_dvecs.append(np.load(dvec_path))
            speakers.append(speaker)
        return speaker_dvecs, speakers

    @staticmethod
    def expand(values, durations):
        out = []
        for value, d in zip(values, durations):
            out += [value] * max(0, int(d))
        return np.array(out)

    def plot(self, sample, show=False):
        if not show:
            matplotlib.use("AGG", force=True)
        mel = sample["mel"]

        if config["dataset"].get("variance_level") == "phoneme":
            pitch = ProcessedDataset.expand(sample["pitch"], sample["duration"])[
                : len(mel)
            ]
            energy = ProcessedDataset.expand(sample["energy"], sample["duration"])[
                : len(mel)
            ]
            if "snr" in sample:
                snr = ProcessedDataset.expand(sample["snr"], sample["duration"])[
                    : len(mel)
                ]
        elif config["dataset"].get("variance_level") == "frame":
            pitch = sample["pitch"][: len(mel)]
            energy = sample["energy"][: len(mel)]
            if "snr" in sample:
                snr = sample["snr"][: len(mel)]

        if str(self.ds.pitch_type) == "pyworld":
            pitch_min, pitch_max = self.stats["pitch_min"], self.stats["pitch_max"]
            pitch = (pitch - pitch_min) / (pitch_max - pitch_min) * mel.shape[1] * 80
        else:
            pitch_min, pitch_max = pitch.min(), pitch.max()
            pitch = (pitch - pitch_min) / (pitch_max - pitch_min) * mel.shape[1]

        energy_min, energy_max = self.stats["energy_min"], self.stats["energy_max"]
        energy = (energy - energy_min) / (energy_max - energy_min) * mel.shape[1] * 80

        if "snr" in sample:
            snr_min, snr_max = -1.1, 1.1
            snr = (snr - snr_min) / (snr_max - snr_min) * mel.shape[1]

        fig = plt.figure(figsize=[6.4 * (len(mel) / 150), 4.8])
        ax = fig.add_subplot()

        ax.imshow(mel.T, origin="lower", cmap="gray")

        last = 0

        for phone, duration in zip(sample["phones"], sample["duration"]):
            x = last
            ax.axline((int(x), 0), (int(x), 80), color="white", alpha=0.3)
            phone = self.id2phone[int(phone)]
            if phone == "[SILENCE]":
                phone = "☐"
            # print(int(x), phone)
            ax.text(int(x), 81, phone)
            last += duration

        if sample["snr"] is None:
            sns.lineplot(
                x=list(range(len(pitch))) + list(range(len(energy))),
                y=list(pitch) + list(energy),
                hue=["Pitch"] * len(pitch) + ["Energy"] * len(energy),
                ax=ax,
                linewidth=2,
            )
        else:
            sns.lineplot(
                x=list(range(len(pitch)))
                + list(range(len(energy)))
                + list(range(len(snr))),
                y=list(pitch) + list(energy) + list(snr),
                hue=["Pitch"] * len(pitch)
                + ["Energy"] * len(energy)
                + ["SNR"] * len(snr),
                ax=ax,
                linewidth=2,
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.ylim(0, 80)
        plt.yticks(range(0, 81, 10))

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
        keys = ["phones", "mel", "pitch", "energy", "duration"]
        if "snr" in data:
            keys.append("snr")
        for key in keys:
            if torch.is_tensor(data[key][0]):
                data[key] = pad_sequence(data[key], batch_first=True, padding_value=0)
            else:
                data[key] = pad_sequence(
                    [torch.tensor(x) for x in data[key]],
                    batch_first=True,
                    padding_value=0,
                )
        if self.ds is not None and self.ds.has_dvector:
            data["speaker"] = torch.stack(data["speaker"])
        else:
            data["speaker"] = torch.tensor(np.array(data["speaker"])).long()
        if self.ds is not None and self.ds.conditioned:
            data["cond_pitch"] = torch.tensor(np.array(data["cond_pitch"]))
            data["cond_energy"] = torch.tensor(np.array(data["cond_energy"]))
            data["cond_duration"] = torch.tensor(np.array(data["cond_duration"]))
            if "snr" in data:
                data["cond_snr"] = torch.tensor(np.array(data["cond_snr"]))
        return data


if __name__ == "__main__":
    train_ud = UnprocessedDataset(
        "../Data/LibriTTS/train-clean-360-aligned", max_entries=10_000, use_snr=True, conditioned=True,
    )
    train = ProcessedDataset(
        unprocessed_ds=train_ud, split="train", phone_vec=False, recompute_stats=True
    )
    #train.plot(train[0], show=True)
