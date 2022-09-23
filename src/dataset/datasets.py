import configparser
import io
import json
import multiprocessing
import os
from glob import glob
from pathlib import Path
from random import Random
import warnings
import shutil

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
from rich import print
from scipy import signal
from third_party.dvectors.wav2mel import Wav2Mel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.rich import tqdm
from tqdm.contrib.concurrent import process_map

from dataset.audio_utils import dynamic_range_compression
from dataset.cwt import CWT
from dataset.snr import SNR

from third_party.argutils import str2bool

pandarallel.initialize(progress_bar=True)
tqdm.pandas()

warnings.filterwarnings("ignore")
np.seterr(divide="raise", invalid="raise")


class TTSDataset(Dataset):
    def __init__(
        self,
        alignments_dataset,
        max_entries=None,
        stat_entries=10_000,
        fmin=0,
        fmax=8000,
        pitch_quality=0.25,
        source_phoneset="arpabet",
        shuffle_seed=42,
        overwrite_stats=False,
        overwrite_stats_if_missing=True,
        _stats=None,
        # provided by model
        speaker_type="dvector",  # "none", "id", "dvector"
        min_length=0.5,
        max_length=32,
        augment_duration=0,  # 0.1,
        variances=["pitch", "energy", "snr"],
        variance_levels=["phone", "phone", "phone"],
        variance_transforms=["cwt", "none", "none"],  # "cwt", "log", "none"
        priors=["pitch", "energy", "snr", "duration"],
        n_mels=80,
        sampling_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        denoise=False,
    ):
        super().__init__()

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
        self.pitch_quality = pitch_quality
        self.do_denoise = denoise

        # PHONES
        self.phone_converter = Converter()

        # TODO: add this upstream
        self.phone_cache = {}

        # AUDIO LOADING
        if isinstance(alignments_dataset, list):
            self.alignment_ds = alignments_dataset
        else:
            self.alignment_ds = [alignments_dataset]

        # DATAFRAME
        self.entry_stats = {
            "missing_textgrids": 0,
            "empty_textgrids": 0,
            "too_short": 0,
            "too_long": 0,
            "bad_transcriptions": 0,
        }
        entries = []
        for i, ds in enumerate(self.alignment_ds):
            entries += [
                entry
                for entry in process_map(
                    self._create_entry,
                    zip([i]*len(ds), np.arange(len(ds))),
                    chunksize=10_000,
                    max_workers=multiprocessing.cpu_count(),
                    desc=f"processing alignments for dataset {i}",
                    tqdm_class=tqdm,
                )
                if entry is not None
            ]
        Random(shuffle_seed).shuffle(entries)

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
            self.speaker2dvector = {}
            for i, row in self.data.iterrows():
                if row["speaker"] not in self.speaker2dvector:
                    self.speaker2dvector[row["speaker"]] = np.load(
                        Path(row["audio"]).parent / "speaker.npy"
                    )
        elif self.speaker_type == "id":
            speakers = self.data["speaker"].unique()
            self.speaker2id = {speaker: i for i, speaker in enumerate(speakers)}

        # DENOISE
        if denoise:
            from speechbrain.pretrained import SpectralMaskEnhancement
            self.denoise = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="speechbrain/metricgan-plus-voicebank",
            )
            self._create_denoised()
            self.data["audio"] = self.data["audio"].apply(lambda x: Path(x).with_suffix(".clean"))
            exist_list = [Path(x).exists() for x in self.data["audio"]]
            print(f"{len(exist_list)-np.sum(exist_list)} missing denoised audio files")
            self.data = self.data[exist_list]

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
        self.phone_level = False

        # COMPUTE STATS
        stat_bs = 4
        stat_list = []
        if stat_entries == None:
            stat_entries = len(self)
        if max_entries == None:
            max_entries = len(self)
        if stat_entries > max_entries:
            stat_entries = max_entries
        stat_path = Path(self.alignment_ds[0].target_directory) / "stats.json"
        compute_stats = True
        if _stats is not None:
            self.stats = _stats
            compute_stats = False
        elif stat_path.exists() and not overwrite_stats:
            with open(stat_path, "r") as f:
                stats = json.load(f)
            for var in self.variances:
                if var not in stats:
                    if overwrite_stats_if_missing:
                        compute_stats = True
                        print(f"missing stats for {var}, overwriting")
                    else:
                        raise ValueError(f"{var} not in stats")
            for prior in self.priors:
                if f"{prior}_prior" not in stats:
                    if overwrite_stats_if_missing:
                        compute_stats = True
                        print(f"missing stats for {prior}, overwriting")
                    else:
                        raise ValueError(f"{prior} not in stats")
            if "samples" in stats and stats["samples"] == stat_entries:
                if "seed" in stats and stats["seed"] == shuffle_seed:
                    print("loading stats from file")
                    compute_stats = False
                    self.stats = stats
                else:
                    raise ValueError("stats file exists but seed does not match")
            else:
                raise ValueError(
                    "stats file exists but number of samples does not match"
                )

        self.speaker2stats = {}
        self.record_speaker_stats = True

        if compute_stats:
            for entry in tqdm(
                DataLoader(
                    self,
                    num_workers=multiprocessing.cpu_count(),
                    batch_size=stat_bs,
                    collate_fn=self._collate_fn,
                    drop_last=True,
                ),
                total=min([stat_entries, max_entries]) // stat_bs,
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
                        std_sq = np.array([s[key][np_stat] for s in stat_list]) ** 2
                        stats[key][np_stat] = float(
                            np.sqrt(np.sum(std_sq) / len(std_sq))
                        )
                    if np_stat == "mean":
                        stats[key][np_stat] = float(
                            np.mean([s[key]["mean"] for s in stat_list])
                        )
                    if np_stat == "min":
                        stats[key][np_stat] = float(
                            np.min([s[key]["min"] for s in stat_list])
                        )
                    if np_stat == "max":
                        stats[key][np_stat] = float(
                            np.max([s[key]["max"] for s in stat_list])
                        )
            stats["seed"] = shuffle_seed
            stats["samples"] = stat_entries
            json.dump(stats, open(stat_path, "w"))
            self.stats = stats

        if "phone" in self.variance_levels:
            self.phone_level = True

    def create_validation_dataset(
        self,
        valid_ds,
        max_entries=None,
        shuffle_seed=42,
    ):
        ds = TTSDataset(
            valid_ds,
            max_entries=max_entries,
            augment_duration=0,
            shuffle_seed=shuffle_seed,
            # use train dataset stats
            _stats=self.stats,
            # use train dataset parameters
            speaker_type=self.speaker_type,
            min_length=self.min_length,
            max_length=self.max_length,
            variances=self.variances,
            variance_levels=self.variance_levels,
            variance_transforms=self.variance_transforms,
            priors=self.priors,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            pitch_quality=self.pitch_quality,
            source_phoneset=self.source_phoneset,
            denoise=self.do_denoise,
        )
        ds.phone2id = self.phone2id
        ds.id2phone = self.id2phone
        ds.record_speaker_stats = False
        return ds

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
        # audio = torch.clip(audio, -1, 1)

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
        silence_ids = [v for k, v in self.phone2id.items() if "[" in k]
        silence_masks = [np.array(row["phones"]) == s for s in silence_ids]
        unexpanded_silence_mask = np.logical_or.reduce(silence_masks)
        silence_mask = TTSDataset._expand(unexpanded_silence_mask, duration)

        # VARIANCES
        # TODO: add variance class
        variances = self._create_variances(audio, silence_mask, duration)

        # PRIORS
        priors = {}
        for var in self.priors:
            if var == "duration":
                priors[var] = np.mean(duration[~unexpanded_silence_mask])
            else:
                if isinstance(variances[var], dict):
                    var_val = variances[var]["original_signal"]
                else:
                    var_val = variances[var]
                mean = 0
                std = 1
                if hasattr(self, "stats"):
                    mean = self.stats[var]["mean"]
                    std = self.stats[var]["std"]
                if (
                    self.variance_levels[self.variances.index(var)] == "phone"
                    and self.phone_level
                ):
                    priors[var] = np.mean(
                        var_val[~unexpanded_silence_mask] * std + mean
                    )
                else:
                    priors[var] = np.mean(var_val[~silence_mask] * std + mean)

        # TEXT & PHONES
        text = row["text"]
        phones = row["phones"]

        # RESULT
        result = {
            "id": item_id,
            "mel": np.array(mel.T)[:dur_sum],
            "variances": {},
            "priors": {},
            "text": text,
            "phones": phones,
            "duration": duration,
            "silence_mask": silence_mask,
            "unexpanded_silence_mask": unexpanded_silence_mask,
        }
        for var in self.variances:
            result["variances"][var] = variances[var]
        for var in self.priors:
            result["priors"][var] = priors[var]
            if row["speaker"] not in self.speaker2stats:
                self.speaker2stats[row["speaker"]] = {var: [] for var in self.priors}
            self.speaker2stats[row["speaker"]][var].append(priors[var])
        if self.speaker_type == "dvector":
            result["speaker"] = self.speaker2dvector[row["speaker"]]
        elif self.speaker_type == "id":
            result["speaker"] = self.speaker2id[row["speaker"]]
        result["speaker_key"] = row["speaker"]

        return result

    def _create_phone2id(self):
        unique_phones = set()
        for phone_list in self.data["phones"]:
            unique_phones.update(phone_list)
        unique_phones = list(sorted(unique_phones))
        phone2id = {p: i + 1 for p, i in zip(unique_phones, range(len(unique_phones)))}
        phone2id["[PAD]"] = 0
        return phone2id

    def _create_variances(self, audio, silence_mask, durations):
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
                variances["pitch"] = variances["pitch"][: sum(durations)]
            variances["pitch"][silence_mask] = np.nan
            if np.isnan(variances["pitch"]).all():
                variances["pitch"][:] = 1e-7
            variances["pitch"] = TTSDataset._interpolate(variances["pitch"])

        # SNR
        if "snr" in self.variances:
            snr = SNR(audio.cpu().numpy().astype(np.float64), self.sampling_rate)
            variances["snr"] = snr.windowed_wada(
                window=self.win_length,
                stride=self.hop_length / self.win_length,
                use_samples=True,
            )
            if len(silence_mask) < len(variances["snr"]):
                variances["snr"] = variances["snr"][: sum(durations)]
            variances["snr"][silence_mask] = np.nan
            if all(np.isnan(variances["snr"])):
                variances["snr"] = np.zeros_like(variances["snr"])
            else:
                variances["snr"] = TTSDataset._interpolate(variances["snr"])

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
                variances["energy"] = variances["energy"][: sum(durations)]

        # TRANSFORMS & Normalize
        for i, var in enumerate(self.variances):
            if self.phone_level and self.variance_levels[i] == "phone":
                pos = 0
                for j, d in enumerate(durations):
                    if d > 0:
                        variances[var][j] = np.mean(variances[var][pos : pos + d])
                    else:
                        variances[var][j] = 1e-7
                    pos += d
                variances[var] = variances[var][: len(durations)]
            if self.variance_transforms[i] == "cwt":
                variances[var] = self.cwt.decompose(variances[var])
            elif self.variance_transforms[i] == "log":
                variances[var] = np.log(variances[var])
            elif hasattr(self, "stats"):
                variances[var] = (
                    variances[var] - self.stats[var]["mean"]
                ) / self.stats[var]["std"]

        return variances

    def _create_denoised(self):
        batch_size = 10
        paths = []
        batch = []
        lengths = []
        for _, row in tqdm(
            self.data.iterrows(),
            desc="creating/loading denoised audio",
            total=len(self.data),
        ):
            denoised_path = row["audio"].with_suffix(".clean.wav")
            if Path(str(denoised_path).replace(".wav", "")).exists():
                continue
            noisy = self.denoise.load_audio(
                str(row["audio"]),
                savedir="speechbrain/wavs"
            )
            paths.append(denoised_path)
            batch.append(noisy)
            lengths.append(len(noisy))
            if len(batch) >= batch_size:
                batch = pad_sequence(
                    batch, batch_first=True, padding_value=0
                )
                clean = self.denoise.enhance_batch(batch, lengths=torch.tensor(lengths)/max(lengths))
                for i, path in enumerate(paths):
                    torchaudio.save(path, clean[i][:lengths[i]].unsqueeze(0).cpu(), 16_000)
                    path.rename(str(path).replace(".wav", ""))
                paths = []
                batch = []
                lengths = []
                shutil.rmtree("speechbrain/wavs")
            

    def _create_dvectors(self):
        wav2mel = Wav2Mel()
        dvector_path = (
            Path(__file__).parent.parent / "third_party" / "dvectors" / "dvector.pt"
        )
        dvector_gen = torch.jit.load(dvector_path).eval()

        for _, row in tqdm(
            self.data.iterrows(),
            desc="creating/loading utt. dvectors",
            total=len(self.data),
        ):
            dvec_path = row["audio"].with_suffix(".npy")
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
            desc="creating/loading speaker dvectors",
        ):
            speaker_df = self.data[self.data["speaker"] == speaker]
            speaker_path = Path(speaker_df.iloc[0]["audio"]).parent / "speaker.npy"
            if not speaker_path.exists():
                dvecs = []
                for _, row in speaker_df.iterrows():
                    dvecs.append(np.load(row["audio"].with_suffix(".npy")))
                dvec = np.mean(dvecs, axis=0)
                np.save(speaker_path, dvec)

    def _create_entry(self, dsi_idx):
        dsi, idx = dsi_idx
        item = self.alignment_ds[dsi][idx]
        start, end = item["phones"][0][0], item["phones"][-1][1]

        if end - start < self.min_length:
            self.entry_stats["too_short"] += 1
            return None
        if end - start > self.max_length:
            self.entry_stats["too_long"] += 1
            return None

        phones = []
        durations = []

        for i, p in enumerate(item["phones"]):
            s, e, phone = p
            phone.replace("ˌ", "")
            r_phone = phone.replace("0", "").replace("1", "")
            if len(r_phone) > 0:
                phone = r_phone
            if "[" not in phone:
                o_phone = phone
                if o_phone not in self.phone_cache:
                    phone = self.phone_converter(
                        phone, self.source_phoneset, lang=None
                    )[0]
                    self.phone_cache[o_phone] = phone
                phone = self.phone_cache[o_phone]
            phones.append(phone)
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        if start >= end:
            self.entry_stats["empty_textgrids"] += 1
            return None

        return (
            phones,
            durations,
            start,
            end,
            item["wav"],
            item["speaker"],
            item["transcript"],
            Path(item["wav"]).name,
        )

    def _create_stats(self, x):
        result = {}
        mel_val = x["mel"]
        mel_val[x["silence_mask"]] = np.nan
        result["mel"] = {}
        result["mel"]["mean"] = torch.nanmean(mel_val)
        result["mel"]["std"] = torch.std(mel_val[~torch.isnan(mel_val)])
        for i, var in enumerate(self.variances):
            if self.variance_transforms[i] == "cwt":
                var_val = x[f"variances_{var}_original_signal"].float()
            else:
                var_val = x[f"variances_{var}"].float()
            var_val[x["silence_mask"]] = np.nan
            result[var] = {}
            result[var]["min"] = torch.min(var_val[~torch.isnan(var_val)])
            result[var]["max"] = torch.max(var_val[~torch.isnan(var_val)])
            result[var]["mean"] = torch.nanmean(var_val)
            result[var]["std"] = torch.std(var_val[~torch.isnan(var_val)])
            if var in self.priors:
                result[var + "_prior"] = {}
                result[var + "_prior"]["min"] = torch.min(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["max"] = torch.max(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["mean"] = torch.mean(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["std"] = torch.std(
                    torch.nanmean(var_val, axis=1)
                )
        for var in self.priors:
            if var not in self.variances:
                var_val = x[var].float()
                var_val[x["unexpanded_silence_mask"]] = np.nan
                result[var + "_prior"] = {}
                result[var + "_prior"]["min"] = torch.min(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["max"] = torch.max(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["mean"] = torch.mean(
                    torch.nanmean(var_val, axis=1)
                )
                result[var + "_prior"]["std"] = torch.std(
                    torch.nanmean(var_val, axis=1)
                )
                del var_val
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
        if isinstance(values, list):
            return out
        elif isinstance(values, torch.Tensor):
            return torch.stack(out)
        elif isinstance(values, np.ndarray):
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
        else:
            for new_key, value in structure.items():
                TTSDataset._flatten(
                    value, new_key, ((path + "_") if path else "") + key, flattened
                )
        return flattened

    def _collate_fn(self, data):
        # list-of-dict -> dict-of-lists
        # (see https://stackoverflow.com/a/33046935)
        data = [TTSDataset._flatten(x) for x in data]
        l2d = lambda x: {k: [dic[k] for dic in x] for k in x[0]}
        data = l2d(data)
        for key in data.keys():
            if isinstance(data[key][0], np.ndarray):
                data[key] = [torch.tensor(x) for x in data[key]]
            if torch.is_tensor(data[key][0]):
                pad_val = 1 if "silence_mask" in key else 0
                data[key] = pad_sequence(
                    data[key], batch_first=True, padding_value=pad_val
                )
        return data

    def plot(self, sample_or_idx, show=True):
        if not show:
            matplotlib.use("AGG", force=True)
        if isinstance(sample_or_idx, int):
            sample = self[sample_or_idx]
        else:
            sample = sample_or_idx

        # MEL SPECTROGRAM
        mel = sample["mel"]
        cwts = self.variance_transforms.count("cwt")
        fig = plt.figure(
            figsize=[7 * (len(mel) / 150) + 3, 4 + 2 * cwts], constrained_layout=True
        )
        gs = GridSpec(5, 4, figure=fig)

        audio_len = len(mel) * self.hop_length / self.sampling_rate
        ax0 = fig.add_subplot(gs[:2, 1:])
        ax0.imshow(
            mel.T,
            origin="lower",
            cmap="gray",
            aspect="auto",
            interpolation="gaussian",
            extent=[0, audio_len, 0, 80],
            alpha=0.8,
        )
        ax0.set_xlim(0, audio_len)
        ax0.set_ylim(0, 80)
        ax0.set_xlabel("Time (seconds)")
        ax0.set_yticks(range(0, 81, 10))
        ax0.set_title(f'"{sample["text"]}"')

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
                var_vals = sample["variances"][var]["original_signal"]
            else:
                var_vals = sample["variances"][var]
            if self.variance_levels[i] == "phone":
                var_vals = TTSDataset._expand(var_vals, sample["duration"])
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
        if num_vars > 0:
            if isinstance(variance_y[0], torch.Tensor):
                variance_y = [x.item() for x in variance_y]
            sns.lineplot(
                x=variance_x,
                y=variance_y,
                hue=variance_text,
                ax=ax0,
                linewidth=2,
            )
        ax_num = 1
        for i, var in enumerate(self.variances):
            cwt_ax = []
            if self.variance_transforms[i] == "cwt":
                spectrogram = sample["variances"][var]["spectrogram"]
                if self.variance_levels[i] == "phone":
                    spectrogram = TTSDataset._expand(spectrogram, sample["duration"])
                cwt_ax.append(fig.add_subplot(gs[1 + ax_num, 1:]))
                cwt_ax[-1].imshow(
                    spectrogram.T,
                    extent=[0, audio_len, 1, 10],
                    cmap="PRGn",
                    aspect="auto",
                    # vmax=1,
                    # vmin=-1,
                    interpolation="gaussian",
                )
                cwt_ax[-1].set_ylabel(f"{var.title()} Frequency")
                cwt_ax[-1].set_ylim(1, 10)
                cwt_ax[-1].set_yticks(range(1, 11, 2))
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

        # SHOW/CREATE IMG
        if show:
            plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.clf()
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    @staticmethod
    def add_model_specific_args(parent_parser, split_name):
        parser = parent_parser.add_argument_group(f"{split_name} Dataset")
        parser.add_argument(f"--{split_name}_max_entries", type=int, default=None)
        parser.add_argument(f"--{split_name}_stat_entries", type=int, default=10_000)
        parser.add_argument(f"--{split_name}_fmin", type=int, default=0)
        parser.add_argument(f"--{split_name}_fmax", type=int, default=8000)
        parser.add_argument(f"--{split_name}_pitch_quality", type=float, default=0.25)
        parser.add_argument(
            f"--{split_name}_source_phoneset", type=str, default="arpabet"
        )
        parser.add_argument(f"--{split_name}_shuffle_seed", type=int, default=42)
        parser.add_argument(f"--{split_name}_overwrite_stats", type=str2bool, default=False)
        parser.add_argument(
            f"--{split_name}_overwrite_stats_if_missing", type=str2bool, default=True
        )
        parser.add_argument(f"--{split_name}_denoise", type=str2bool, default=False)
        return parent_parser
