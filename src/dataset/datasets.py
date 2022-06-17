import configparser
import io
import json
import multiprocessing
import os
from glob import glob
from pathlib import Path
from random import Random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyworld as pw
import scipy
import scipy.stats as stats
import seaborn as sns
import tgt
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as AT
import torchvision.transforms as VT
from librosa.filters import mel as librosa_mel
from matplotlib.gridspec import GridSpec
from pandarallel import pandarallel
from phones.convert import Converter
from PIL import Image
from scipy import signal
from third_party.dvectors.wav2mel import Wav2Mel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

from dataset.audio_utils import dynamic_range_compression, get_alignment
from dataset.cwt import CWT
from dataset.snr import SNR

pandarallel.initialize(progress_bar=True)
tqdm.pandas()

np.seterr(divide="raise", invalid="raise")


class UnprocessedDataset(Dataset):
    def __init__(
        self,
        audio_directory,  # can be several as well
        alignment_directory=None,  # can be several as well
        max_entries=None,
        stat_entries=10_000,
        speaker_type="dvector",  # "none", "id", "dvector"
        min_length=0.5,
        max_length=32,
        augment_duration=0, # 0.1,
        variances=["pitch", "energy", "snr"],
        variance_levels=["phone", "phone", "phone"],
        variance_transforms=["cwt", "none", "none"],  # "cwt", "log", "none"
        priors=["pitch", "energy", "snr", "duration"],
        sampling_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        fmin=0,
        fmax=8000,
        pitch_quality=0.25,
        source_phoneset="arpabet",
        shuffle_seed=42,
    ):
        super().__init__()

        self.dir = audio_directory
        if alignment_directory is None:
            alignment_directory = audio_directory

        # HPARAMS
        self.min_length = min_length
        self.max_length = max_length
        self.max_frames = int(np.ceil(sampling_rate * max_length / hop_length))
        self.augment_duration = augment_duration
        self.speaker_type = speaker_type
        self.variances = variances
        self.variance_levels = variance_levels
        self.priors = priors
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.dio_speed = int(np.round(1 / pitch_quality))
        self.source_phoneset = source_phoneset

        # PHONES
        self.phone_converter = Converter()

        # TODO: add this upstream
        self.phone_cache = {}

        # TEXTGRID LOADING
        if not isinstance(alignment_directory, list):
            alignment_directory = [alignment_directory]
        self.grid_files = {}
        for ali_dir in alignment_directory:
            for file in glob(os.path.join(ali_dir, "**/*.TextGrid"), recursive=True):
                self.grid_files[Path(file).name.replace(".TextGrid", ".wav")] = file

        # AUDIO LOADING
        if not isinstance(audio_directory, list):
            audio_directory = [audio_directory]
        entry_list = []
        for aud_dir in audio_directory:
            entry_list += list(glob(os.path.join(aud_dir, "**", "*.wav")))
        if max_entries is not None:
            entry_list = entry_list[:max_entries]
        Random(shuffle_seed).shuffle(entry_list)

        # DATAFRAME
        self.entry_stats = {
            "missing_textgrids": 0,
            "empty_textgrids": 0,
            "too_short": 0,
            "too_long": 0,
            "bad_transcriptions": 0,
        }
        entries = [
            entry
            for entry in process_map(
                self._create_entry,
                entry_list,
                chunksize=100,
                max_workers=multiprocessing.cpu_count(),
                desc="collecting textgrid and audio files",
            )
            if entry is not None
        ]
        print("data loading stats:")
        for key in self.entry_stats:
            print(f"{key}: {self.entry_stats[key]}")
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

        # DVECTORS
        if self.speaker_type == "dvector":
            self._create_dvectors()
            dvecs = []
            for i, row in self.data.iterrows():
                dvecs.append(Path(row["audio"]).parent / "speaker.npy")
            self.data["dvector"] = dvecs

        # MEL SPECTROGRAM
        self.mel_spectrogram = AT.Spectrogram(
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
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()

        # VARIANCE TRANSFORMS
        self.variance_transforms = variance_transforms
        if "cwt" in self.variance_transforms:
            self.cwt = CWT()

        # PHONES TO IDS
        self.phone2id = self._create_phone2id()
        self.id2phone = {v: k for k, v in self.phone2id.items()}
        self.data["phones"] = self.data["phones"].apply(
            lambda x: torch.tensor([self.phone2id[p] for p in x]).long()
        )

        # COMPUTE STATS
        stat_bs = 16
        stat_list = []
        for entry in tqdm(
            DataLoader(
                self,
                num_workers=multiprocessing.cpu_count(),
                batch_size=stat_bs,
                collate_fn=self._collate_fn,
                drop_last=True,
            ),
            total=min([stat_entries,max_entries])//stat_bs,
            desc="computing stats",
        ):
            if len(stat_list) * stat_bs >= stat_entries:
                break
            stat_list.append(self._create_stats(entry))
        stats = {}
        stats["sample_size"] = stat_entries
        for key in stat_list[0].keys():
            stats[key] = {}
            for np_stat in stat_list[0][key].keys():
                if np_stat == "std":
                    std_sq = np.array([s[key][np_stat] for s in stat_list])**2
                    mean_list = np.array([s[key]["mean"] for s in stat_list])
                    mean_diff_sq = (mean_list - np.mean(mean_list))**2
                    stats[key][np_stat] = np.sqrt(np.mean(std_sq+mean_diff_sq))
                else:
                    stats[key][np_stat] = getattr(np, np_stat)(
                        [s[key][np_stat] for s in stat_list]
                    )
        self.stats = stats
        print(stats)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ID
        item_id = row["basename"]

        # LOAD AUDIO
        audio, sampling_rate = torchaudio.load(row["audio"])
        if sampling_rate != self.sampling_rate:
            transform = AT.Resample(sampling_rate, self.sampling_rate)
            audio = transform(audio)
        start = int(self.sampling_rate * row["start"])
        end = int(self.sampling_rate * row["end"])
        audio = audio[0][start:end]
        audio = audio / torch.max(torch.abs(audio))

        # MEL SPECTROGRAM
        mel = self.mel_spectrogram(audio.unsqueeze(0))
        mel = torch.sqrt(mel[0])
        mel = torch.matmul(self.mel_basis, mel)
        mel = dynamic_range_compression(mel).cpu()

        # DURATIONS & SILENCE MASK
        duration = np.array(row["duration"])
        if self.augment_duration > 0:
            duration = self._augment_duration(duration)
        dur_sum = sum(duration)
        unexpanded_silence_mask = np.array(row["phones"]) == self.phone2id["[SILENCE]"]
        silence_mask = UnprocessedDataset._expand(unexpanded_silence_mask, duration)

        # VARIANCES
        # TODO: add variance class
        variances = self._create_variances(audio, silence_mask)

        # PRIORS
        priors = {}
        for var in self.priors:
            if var == "duration":
                priors[var] = np.mean(duration[~unexpanded_silence_mask])
            else:
                if isinstance(variances[var], dict):
                    var_val = variances[var]["signal"]
                else:
                    var_val = variances[var]
                priors[var] = np.mean(var_val[~silence_mask])

        # TEXT & PHONES
        text = row["text"]
        phones = row["phones"]

        # RESULT
        result = {
            "id": item_id,
            "mel": np.array(mel.T)[:dur_sum],
            "variances": {},
            "priors": {},
            "speaker": {},
            "text": text,
            "phones": phones,
            "duration": duration,
        }
        for var in self.variances:
            result["variances"][var] = variances[var]
        for var in self.priors:
            result["priors"][var] = priors[var]
        result["speaker"]["id"] = row["speaker"]
        if self.speaker_type == "dvector":
            result["speaker"]["dvector"] = str(row["dvector"])

        return result

    def _create_phone2id(self):
        unique_phones = set()
        for phone_list in self.data["phones"]:
            unique_phones.update(phone_list)
        unique_phones = list(sorted(unique_phones))
        phone2id = {p: i + 1 for p, i in zip(unique_phones, range(len(unique_phones)))}
        phone2id["[PAD]"] = 0
        return phone2id

    def _create_variances(self, audio, silence_mask):
        variances = {}

        # PITCH
        if "pitch" in self.variances:
            pitch, t = pw.dio(
                audio.cpu().numpy().astype(np.float64),
                self.sampling_rate,
                frame_period=self.hop_length / self.sampling_rate * 1000,
                speed=self.dio_speed,
            )
            variances["pitch"] = pw.stonemask(
                audio.cpu().numpy().astype(np.float64), pitch, t, self.sampling_rate
            )
            variances["pitch"][variances["pitch"] == 0] = np.nan
            if len(silence_mask) < len(variances["pitch"]):
                variances["pitch"] = variances["pitch"][:len(silence_mask)]
            variances["pitch"][silence_mask] = np.nan
            if np.isnan(variances["pitch"]).all():
                variances["pitch"][:] = 1e-7
                print(variances["pitch"])
            variances["pitch"] = UnprocessedDataset._interpolate(variances["pitch"])

        # SNR
        if "snr" in self.variances:
            snr = SNR(audio.cpu().numpy().astype(np.float64), self.sampling_rate)
            variances["snr"] = snr.windowed_wada(
                window=self.win_length,
                stride=self.hop_length / self.win_length,
                use_samples=True,
            )
            if len(silence_mask) < len(variances["snr"]):
                variances["snr"] = variances["snr"][:len(silence_mask)]
            variances["snr"][silence_mask] = np.nan
            if all(np.isnan(variances["snr"])):
                variances["snr"] = np.zeros_like(variances["snr"])
            else:
                variances["snr"] = UnprocessedDataset._interpolate(variances["snr"])

        # ENERGY
        if "energy" in self.variances:
            variances["energy"] = np.array(
                [
                    np.sqrt(
                        np.sum(
                            (
                                audio[
                                    x * self.hop_length : (x * self.hop_length)
                                    + self.win_length
                                ]
                                ** 2
                            ).numpy()
                        )
                        / self.win_length
                    )
                    for x in range(int(np.ceil(len(audio) / self.hop_length)))
                ]
            )
            if len(silence_mask) < len(variances["energy"]):
                variances["energy"] = variances["energy"][:len(silence_mask)]

        # TRANSFORMS
        for i, var in enumerate(self.variances):
            if self.variance_transforms[i] == "cwt":
                variances[var] = self.cwt.decompose(variances[var])
            if self.variance_transforms[i] == "log":
                variances[var] = np.log(variances[var])

        return variances

    def _create_dvectors(self):
        wav2mel = Wav2Mel()
        dvector_path = (
            Path(__file__).parent.parent / "third_party" / "dvectors" / "dvector.pt"
        )
        dvector_gen = torch.jit.load(dvector_path).eval()

        for _, row in tqdm(
            self.data.iterrows(),
            desc="creating/loading utt. dvectors (this only has to be done once)",
            total=len(self.data),
        ):
            dvec_path = Path(row["audio"].replace(".wav", ".npy"))
            if not dvec_path.exists():
                audio, sampling_rate = torchaudio.load(row["audio"])
                start = int(self.sampling_rate * row["start"])
                end = int(self.sampling_rate * (row["start"] + 1))
                audio = audio[0][start:end]
                audio = audio / torch.max(torch.abs(audio))  # might not be necessary

                if 16_000 != self.sampling_rate:
                    transform = AT.Resample(sampling_rate, self.sampling_rate)
                    audio = transform(audio)
                dvector_mel = wav2mel(torch.unsqueeze(audio, 0).cpu(), 16_000)
                dvec_result = dvector_gen.embed_utterance(dvector_mel).detach()
                np.save(dvec_path, dvec_result.numpy())

        for speaker in tqdm(
            self.data["speaker"].unique(),
            desc="creating/loading speaker dvectors (this only has to be done once)",
        ):
            speaker_df = self.data[self.data["speaker"] == speaker]
            speaker_path = Path(speaker_df.iloc[0]["audio"]).parent / "speaker.npy"
            if not speaker_path.exists():
                dvecs = []
                for _, row in speaker_df.iterrows():
                    dvecs.append(np.load(row["audio"].replace(".wav", ".npy")))
                dvec = np.mean(dvecs, axis=0)
                np.save(speaker_path, dvec)

    def _create_entry(self, audio_file):
        extract_speaker = lambda x: x.split("/")[-2]
        audio_name = Path(audio_file).name

        try:
            grid_file = self.grid_files[audio_name]
            textgrid = tgt.io.read_textgrid(grid_file)
        except KeyError:
            self.entry_stats["missing_textgrids"] += 1
            return None
        ali = get_alignment(
            textgrid.get_tier_by_name("phones"),
            self.sampling_rate,
            self.hop_length,
        )

        phones, durations, start, end = ali

        if end - start < self.min_length:
            self.entry_stats["too_short"] += 1
            return None
        if end - start > self.max_length:
            self.entry_stats["too_long"] += 1
            return None

        for i, phone in enumerate(phones):
            add_stress = False
            phone.replace("ˌ", "")
            r_phone = phone.replace("0", "").replace("1", "")
            if len(r_phone) > 0:
                phone = r_phone
            if phone not in ["spn", "sil"]:
                o_phone = phone
                if o_phone not in self.phone_cache:
                    phone = self.phone_converter(
                        phone, self.source_phoneset, lang=None
                    )[0]
                    self.phone_cache[o_phone] = phone
                phone = self.phone_cache[o_phone]
                if add_stress:
                    phone = "ˈ" + phone
            else:
                phone = "[SILENCE]"
            phones[i] = phone
        if start >= end:
            self.entry_stats["empty_textgrids"] += 1
            return None

        try:
            with open(audio_file.replace(".wav", ".lab"), "r") as f:
                transcription = f.read()
        except UnicodeDecodeError:
            self.entry_stats["bad_transcriptions"] += 1
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

    def _create_stats(self, x):
        result = {}
        for i, var in enumerate(self.variances):
            if self.variance_transforms[i] == "cwt":
                var_val = x[f"variances_{var}_signal"].float()
            else:
                var_val = x[f"variances_{var}"].float()
            result[var] = {}
            result[var]["min"] = torch.min(var_val)
            result[var]["max"] = torch.max(var_val)
            result[var]["mean"] = torch.mean(var_val)
            result[var]["std"] = torch.std(var_val)
            if var in self.priors:
                result[var + "_prior"] = {}
                result[var + "_prior"]["min"] = torch.min(torch.mean(var_val, axis=0))
                result[var + "_prior"]["max"] = torch.max(torch.mean(var_val, axis=0))
                result[var + "_prior"]["mean"] = torch.mean(torch.mean(var_val, axis=0))
                result[var + "_prior"]["std"] = torch.std(torch.mean(var_val, axis=0))
        for var in self.priors:
            if var not in self.variances:
                var_val = x[var].float()
                result[var + "_prior"] = {}
                result[var + "_prior"]["min"] = torch.min(torch.mean(var_val, axis=0))
                result[var + "_prior"]["max"] = torch.max(torch.mean(var_val, axis=0))
                result[var + "_prior"]["mean"] = torch.mean(torch.mean(var_val, axis=0))
                result[var + "_prior"]["std"] = torch.std(torch.mean(var_val, axis=0))
        return result

    def _augment_duration(self, duration):
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
        return duration

    @staticmethod
    def _expand(values, durations):
        out = []
        for value, d in zip(values, durations):
            out += [value] * max(0, int(d))
        return np.array(out)

    @staticmethod
    def _interpolate(x):
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        nans, y = nan_helper(x)
        x[nans] = np.interp(y(nans), y(~nans), x[~nans])
        return x

    @staticmethod
    def _flatten(structure, key="", path="", flattened=None):
        if flattened is None:
            flattened = {}
        if not isinstance(structure, dict):
            flattened[((path + "_") if path else "") + key] = structure
        # elif isinstance(structure, list):
        #     for i, item in enumerate(structure):
        #         UnprocessedDataset._flatten(item, "%d" % i, path + "_" + key, flattened)
        else:
            for new_key, value in structure.items():
                UnprocessedDataset._flatten(
                    value, new_key, ((path + "_") if path else "") + key, flattened
                )
        return flattened

    def _collate_fn(self, data):
        # list-of-dict -> dict-of-lists
        # (see https://stackoverflow.com/a/33046935)
        data = [UnprocessedDataset._flatten(x) for x in data]
        l2d = lambda x: {k: [dic[k] for dic in x] for k in x[0]}
        data = l2d(data)
        for key in data.keys():
            if isinstance(data[key][0], np.ndarray):
                data[key] = [torch.tensor(x) for x in data[key]]
            if torch.is_tensor(data[key][0]):
                data[key] = pad_sequence(data[key], batch_first=True, padding_value=0)
        return data

    def plot(self, sample_or_idx, show=True):
        if not show:
            matplotlib.use("AGG", force=True)
        if isinstance(sample_or_idx, int):
            sample = self[sample_or_idx]

        # MEL SPECTROGRAM
        mel = sample["mel"]
        cwts = self.variance_transforms.count("cwt")
        fig = plt.figure(
            figsize=[7 * (len(mel) / 150) + 3, 4 + 2 * cwts],
            constrained_layout=True
        )
        gs = GridSpec(4, 4, figure=fig)
        
        audio_len = len(mel) * self.hop_length / self.sampling_rate
        ax0 = fig.add_subplot(gs[:2, 1:])
        ax0.imshow(
            mel.T,
            origin="lower",
            cmap="gray",
            aspect="auto",
            interpolation="gaussian",
            extent=[0, audio_len, 0, 80],
            alpha=0.5,
        )
        ax0.set_xlim(0, audio_len)
        ax0.set_ylim(0, 80)
        ax0.set_xlabel("Time (seconds)")
        ax0.set_yticks(range(0, 81, 10))

        # PHONES
        x = 0
        ax2 = ax0.twiny()
        phone_x = []
        phone_l = []
        for phone, duration in zip(sample["phones"], sample["duration"]):
            phone = self.id2phone[phone.item()]
            new_x = x * self.hop_length / self.sampling_rate
            ax0.axline((new_x, 0), (new_x, 80), color="white", alpha=0.3)
            if phone == "[SILENCE]":
                phone = "☐"
            phone_x.append(new_x + duration * self.hop_length / self.sampling_rate / 2)
            phone_l.append(phone)
            x += duration
        ax2.set_xlim(0, audio_len)
        ax2.set_xticks(phone_x)
        ax2.set_xticklabels(phone_l)
        ax2.set_xlabel("Phones (IPA)")
        ax2.tick_params(axis="x", labelsize=8)

        # VARIANCES
        variance_text = []
        variance_x = []
        variance_y = []
        num_vars = len(self.variances)
        for i, var in enumerate(self.variances):
            if self.variance_transforms[i] == "cwt":
                var_vals = sample["variances"][var]["signal"]
            else:
                var_vals = sample["variances"][var]
            variance_text += [var] * len(mel)
            variance_x += list(
                np.array(range(len(mel))) * self.hop_length / self.sampling_rate
            )
            variance_y += list(
                (var_vals - var_vals.min())
                / ((var_vals.max() - var_vals.min()) + 1e-7)
                * (80 / num_vars)
                + (i * 80 / num_vars)
            )
        sns.lineplot(
            x=variance_x,
            y=variance_y,
            hue=variance_text,
            ax=ax0,
            linewidth=2,
        )
        for i, var in enumerate(self.variances):
            ax_num = 1
            if self.variance_transforms[i] == "cwt":
                spectrogram = sample["variances"][var]["spectrogram"]
                cwt_ax = fig.add_subplot(gs[1+ax_num, 1:], sharex=ax0)
                cwt_ax.imshow(
                    spectrogram.T,
                    extent=[0, audio_len, 1, 10],
                    cmap="PRGn",
                    aspect="auto",
                    vmax=abs(spectrogram).max(),
                    vmin=-abs(spectrogram).max(),
                    interpolation="gaussian",
                )
                cwt_ax.set_ylabel(f"{var.title()} Frequency")
                cwt_ax.set_ylim(1,10)
                cwt_ax.set_yticks(range(1,11,2))
                ax_num += 1

        # PRIORS
        for i, var in enumerate(self.priors):
            prior_ax = fig.add_subplot(gs[i, 0])
            prior_vals = self.stats[f"{var}_prior"]
            mu, sig = prior_vals["mean"], prior_vals["std"]
            pmin, pmax = prior_vals["min"], prior_vals["max"]
            x = np.linspace(pmin, pmax)
            prior_ax.plot(x, stats.norm.pdf(x, mu, sig))
            prior_ax.set_title(f"{var} Prior")
            prior_ax.axvline(sample["priors"][var], color="red")

        # LEGEND & AXES
        plt.tight_layout()

        # SHOW/CREATE IMG
        if show:
            plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.clf()
        buf.seek(0)
        plt.close()
        return Image.open(buf)


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

        return sample

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

        fig = plt.figure(figsize=[6.4 * (len(mel) / 150), 4.8], constrained_layout=True)
        ax = fig.add_subplot()

        ax.imshow(mel.T, origin="lower", cmap="gray", interpolation="gaussian")

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
