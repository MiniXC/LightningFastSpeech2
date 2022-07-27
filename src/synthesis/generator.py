from audioop import mul
from pathlib import Path
import random
import shutil
import multiprocessing
import pickle

import torch
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

from third_party.hifigan import Synthesiser
from fastspeech2.fastspeech2 import FastSpeech2
from .g2p import G2P
from copy import deepcopy


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


class SpeechGenerator:
    def __init__(
        self,
        model_path: str,
        g2p_model: G2P,
        device: str = "cuda:0",
        synth_device: str = None,
        overwrite: bool = False,
        sampling_path: str = None,
        augmentations = None,
        speaker_dict = None,
    ):
        self.model_path = model_path
        if synth_device is None:
            self.synth = Synthesiser(device=device)
        else:
            self.synth = Synthesiser(device=synth_device)
        self.model = FastSpeech2.load_from_checkpoint(self.model_path)
        self.model.eval()
        self.g2p = g2p_model
        self.device = device
        self.model.to(self.device)
        self.overwrite = overwrite
        self.sampling_path = sampling_path
        self.augmentations = augmentations
        self.speaker_dict = speaker_dict

    @property
    def speakers(self):
        if self.model.hparams.speaker_type == "dvector":
            return self.model.speaker2dvector.keys()
        elif self.model.hparams.speaker_type == "id":
            return self.model.speaker2id.keys()
        else:
            return None

    def generate_sample_from_text(self, text, speaker=None):
        ids = [
            self.model.phone2id[x] for x in self.g2p(text) if x in self.model.phone2id
        ]
        batch = {}
        if self.model.hparams.speaker_type == "dvector":
            if speaker is None:
                speaker = list(self.model.speaker2dvector.keys())[
                    np.random.randint(len(self.model.speaker2dvector))
                ]
                print("Using speaker", speaker)
            batch["speaker"] = torch.tensor([self.model.speaker2dvector[speaker]]).to(
                self.device
            )
        if self.model.hparams.speaker_type == "id":
            batch["speaker"] = torch.tensor([self.model.speaker2id[speaker]]).to(
                self.device
            )
        batch["phones"] = torch.tensor([ids]).to(self.device)
        return self.generate_samples(batch)[0]

    def generate_samples(
        self,
        batch,
        increase_diversity={},
        fixed_diversity={},
        sampling_diversity={},
        oracle_diversity={},
        prior_sampling="none", # "none", "normal"
    ):
        if len(self.model.hparams.priors) > 0:
            if prior_sampling == "none":
                raise ValueError("Prior sampling has to be enabled")
            for p in self.model.hparams.priors:
                if prior_sampling == "normal":
                    p_mean = self.model.stats[f"{p}_prior"]["mean"]
                    p_std = self.model.stats[f"{p}_prior"]["std"]
                    p_rand = np.random.normal(p_mean, p_std, size=len(batch[f'priors_{p}']))
                    batch[f'priors_{p}'] = p_rand
                if prior_sampling == "uniform":
                    p_min = self.model.stats[f"{p}_prior"]["min"]
                    p_max = self.model.stats[f"{p}_prior"]["max"]
                    p_rand = np.random.uniform(p_min, p_max, size=len(batch[f'priors_{p}']))
                    batch[f'priors_{p}'] = p_rand
                if prior_sampling == "oracle":
                    pass
                if prior_sampling == "speaker":
                    if self.speaker_dict is None:
                        raise ValueError("Speaker dict not provided")
                    for i, sk in enumerate(batch["speaker_key"]):
                        batch[f'priors_{p}'][i] = random.choice()
                if prior_sampling == "speaker_uniform":
                    if self.speaker_dict is None:
                        raise ValueError("Speaker dict not provided")
                    for i, sk in enumerate(batch["speaker_key"]):
                        s_min = np.min(self.speaker_dict[p][sk])
                        s_max = np.max(self.speaker_dict[p][sk])
                        batch[f'priors_{p}'][i] = random.uniform(s_min, s_max)
                if prior_sampling == "speaker_normal":
                    if self.speaker_dict is None:
                        raise ValueError("Speaker dict not provided")
                    for i, sk in enumerate(batch["speaker_key"]):
                        s_mean = np.mean(self.speaker_dict[p][sk])
                        s_std = np.std(self.speaker_dict[p][sk])
                        s_min = np.min(self.speaker_dict[p][sk])
                        s_max = np.max(self.speaker_dict[p][sk])
                        batch[f'priors_{p}'][i] = np.clip(np.random.normal(s_mean, s_std), s_min, s_max)

        result = self.model(batch, inference=True)

        changed_diversity = (
            len(increase_diversity) > 0
            or len(fixed_diversity) > 0
            or len(sampling_diversity) > 0
        )

        if changed_diversity and any(oracle_diversity.values()):
            raise ValueError(
                "Cannot use oracle diversity when changing diversity"
            )

        if changed_diversity and not any(oracle_diversity.values()):
            for var in self.model.hparams.variances:
                batch["variances_" + var] = result["variances_" + var]
            batch["duration"] = result["duration_rounded"]

        # random factor
        if len(increase_diversity) > 0:
            for key, value in increase_diversity.items():
                if key == "duration":
                    batch[key] = result[key + "_rounded"].float() * np.random.uniform(
                        1.0 - value / 2, 1.0 + value / 2
                    )
                    batch[key] = torch.round(batch[key]).int()
                else:
                    batch[key] = result[key].float() * np.random.uniform(
                        1.0 - value / 2, 1.0 + value / 2
                    )

        # fixed value
        if len(fixed_diversity) > 0:
            for key, value in fixed_diversity.items():
                if key == "duration":
                    batch[key][:] = (
                        value - self.model.stats[key]["mean"]
                    ) / self.model.stats[key]["std"]
                else:
                    stat_key = key.replace("variances_", "")
                    batch[key][:] = (
                        value - self.model.stats[stat_key]["mean"]
                    ) / self.model.stats[stat_key]["std"]

        # sampling
        if len(sampling_diversity) > 0:
            for key, value in sampling_diversity.items():
                rand_vals = np.random.uniform(0, 1, batch[key].shape)
                sampling_mask = (rand_vals < value) & ~batch["phones"].eq(0).numpy()
                sampling_len = np.sum(sampling_mask)
                sampling_mask = torch.tensor(sampling_mask)
                samples = np.random.choice(
                    self.sampling_dict[key], sampling_len, replace=False
                )
                batch[key][sampling_mask] = (
                    torch.tensor(samples).to(batch[key].dtype).to(batch[key].device)
                )

        if any(oracle_diversity.values()):
            for key, value in oracle_diversity.items():
                if value:
                    pass
                else:
                    batch[key] = result[key].to(self.device)
            for key in batch.keys():
                try:
                    batch[key] = batch[key].to(self.device)
                except:
                    pass
            result = self.model(batch, inference=False)

        if changed_diversity:
            for key in batch.keys():
                try:
                    batch[key] = batch[key].to(self.device)
                except:
                    pass
            result = self.model(batch, inference=False)
        
        mels = []
        for i in range(len(result["mel"])):
            if changed_diversity or ("duration" in oracle_diversity and oracle_diversity["duration"]):
                pred_mel = result["mel"][i][: torch.sum(batch["duration"][i])].cpu()
            else:
                pred_mel = result["mel"][i][
                    : torch.sum(result["duration_rounded"][i])
                ].cpu()
            mels.append(int16_samples_to_float32(self.synth(pred_mel)[0]))
        if self.augmentations is not None:
            mels = [self.augmentations(m, sample_rate=self.model.hparams.sampling_rate) for m in mels]
        return mels

    def _create_phone_sampling_dict(self, dataset, variances, batch_size): # refactor for DRYness
        self.sampling_dict = {}
        done_vars = []
        if self.sampling_path is not None:
            for key in variances:
                dict_path = Path(self.sampling_path, key + "_phone" + ".pkl")
                if dict_path.exists():
                    self.sampling_dict[key] = pickle.load(open(dict_path, "rb"))
        if all([k in self.sampling_dict for k in variances]):
            return
        else:
            for k in self.sampling_dict:
                done_vars.append(k)
        for item in tqdm(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset._collate_fn,
                num_workers=multiprocessing.cpu_count(),
            ),
            desc="Creating sampling dict",
        ):
            for key, value in item.items():
                if (
                    key not in done_vars
                    and key in variances
                    and (
                        key not in self.sampling_dict
                        or len(self.sampling_dict[key]) < 1_000_000
                    )
                ):
                    if key == "duration":
                        var_lvl = "phone"
                    else:
                        var_key = key.replace("variances_", "")
                        var_idx = self.model.hparams.variances.index(var_key)
                        var_lvl = self.model.hparams.variance_levels[var_idx]
                    if var_lvl == "phone":
                        for phone in self.model.phone2id.keys():
                            if phone != "[PAD]":
                                if phone not in self.sampling_dict[key]:
                                    self.sampling_dict[key][phone] = []
                                var_vals = (
                                    value[item["phones"].eq(self.model.phone2id[phone])]
                                    .cpu()
                                    .numpy()
                                    .flatten()
                                    .tolist()
                                )
                            self.sampling_dict[key][phone] += var_vals
                    elif var_lvl == "frame":
                        # not supported yet
                        print("Frame variance not supported yet")
        for key in self.sampling_dict.keys():
            # to numpy arrays
            self.sampling_dict[key] = { k: np.array(v) for k, v in self.sampling_dict[key].items() }
            if self.sampling_path is not None:
                Path(self.sampling_path).mkdir(parents=True, exist_ok=True)
                dict_path = Path(self.sampling_path, key + "_phone" + ".pkl")
                pickle.dump(self.sampling_dict[key], open(dict_path, "wb"))

    def _create_sampling_dict(self, dataset, variances, batch_size):
        self.sampling_dict = {}
        done_vars = []
        if self.sampling_path is not None:
            for key in variances:
                if Path(self.sampling_path, key + ".npy").exists():
                    self.sampling_dict[key] = np.load(
                        self.sampling_path + "/" + key + ".npy"
                    )
        if all([k in self.sampling_dict for k in variances]):
            return
        else:
            for k in self.sampling_dict:
                done_vars.append(k)
        for item in tqdm(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset._collate_fn,
                num_workers=multiprocessing.cpu_count(),
            ),
            desc="Creating sampling dict",
        ):
            for key, value in item.items():
                if (
                    key not in done_vars
                    and key in variances
                    and (
                        key not in self.sampling_dict
                        or len(self.sampling_dict[key]) < 1_000_000
                    )
                ):
                    if key == "duration":
                        var_lvl = "phone"
                    else:
                        var_key = key.replace("variances_", "")
                        var_idx = self.model.hparams.variances.index(var_key)
                        var_lvl = self.model.hparams.variance_levels[var_idx]
                    if var_lvl == "phone":
                        var_vals = (
                            value[~item["phones"].eq(0)]
                            .cpu()
                            .numpy()
                            .flatten()
                            .tolist()
                        )
                    elif var_lvl == "frame":
                        var_vals = (
                            value[~item["tgt_mask"]].cpu().numpy().flatten().tolist()
                        )
                    if key not in self.sampling_dict:
                        self.sampling_dict[key] = var_vals
        for key in self.sampling_dict.keys():
            self.sampling_dict[key] = np.array(self.sampling_dict[key])
            if self.sampling_path is not None:
                Path(self.sampling_path).mkdir(parents=True, exist_ok=True)
                np.save(
                    self.sampling_path + "/" + key + ".npy", self.sampling_dict[key]
                )

    def _create_dataset2model(self, dataset, dataset_dvectors, model_dvectors):
        if len(dataset.speaker2dvector) > len(self.model.speaker2dvector):
            model2dataset = {}
            for m_id, m_speaker in model_dvectors.items():
                closest_dist = float("inf")
                closest_speaker = None
                for d_id, d_speaker in dataset_dvectors.items():
                    dist = np.sum(np.abs(np.array(m_speaker) - np.array(d_speaker)))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_speaker = d_id
                model2dataset[m_id] = closest_speaker
                del dataset_dvectors[closest_speaker]
            dataset2model = {v: k for k, v in model2dataset.items()}
            print(
                "WARNING: There are more speakers in the dataset than in the model, this means that some speakers will be picked randomly"
            )
        else:
            dataset2model = {}
            for d_id, d_speaker in dataset_dvectors.items():
                closest_dist = float("inf")
                closest_speaker = None
                for m_id, m_speaker in model_dvectors.items():
                    dist = np.sum(np.abs(np.array(m_speaker) - np.array(d_speaker)))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_speaker = m_id
                dataset2model[d_id] = closest_speaker
                del model_dvectors[closest_speaker]
        return dataset2model

    def generate_from_dataset(
        self,
        dataset,
        target_dir,
        hours=20,
        batch_size=6,
        increase_diversity={},
        fixed_diversity={},
        sampling_diversity={},
        oracle_diversity={},
        sampling_level="all",
        filter_speakers=None, # reduces the number of speakers to the top x with the most samples available
        copy=False, # simply copies the files to the target directory
        random_speaker=False, # randomly selects a speaker from the tts model
        include_dataset_speakers=False, # includes the speakers from the dataset in the random speaker selection
        prior_sampling="none", # prior sampling
    ):
        dataset.stats = self.model.stats
        if Path(target_dir).exists() and not self.overwrite:
            print("Target directory exists, not overwriting")
            return
        else:
            shutil.rmtree(target_dir, ignore_errors=True)
        Path(target_dir).mkdir(parents=True, exist_ok=False)
        if self.model.hparams.speaker_type == "dvector":
            dataset_dvectors = deepcopy(dataset.speaker2dvector)
            model_dvectors = deepcopy(self.model.speaker2dvector)
            print(
                f"Dataset has {len(dataset_dvectors)} speakers, model has {len(model_dvectors)}"
            )
            if random_speaker and copy:
                print("Random speaker and copy, this is not supported")
                return
            if not random_speaker:
                dataset2model = self._create_dataset2model(dataset, dataset_dvectors, model_dvectors)
            if include_dataset_speakers and not random_speaker:
                print("Including dataset speakers in random speaker selection, this is not supported")
                return
            else:
                if include_dataset_speakers:
                    random_vecs = {**dataset_dvectors, **model_dvectors}
                else:
                    random_vecs = model_dvectors
                if filter_speakers is not None:
                    random_vecs = {k: v for k, v in list(random_vecs.items())[:filter_speakers]}
                    random_keys = sorted(random_vecs.keys())
                    random_weights = {k: 0.01 for k in random_keys}
                    print(f"Filtered dataset down to {filter_speakers} speakers")
            pbar = tqdm(total=hours, desc="Generating Audio")
            total_hours = 0
            np.random.seed(42)
            if len(sampling_diversity) > 0:
                if sampling_level == "all":
                    self._create_sampling_dict(
                        dataset, sampling_diversity.keys(), batch_size * 10
                    )
                elif sampling_level == "phone":
                    self._create_phone_sampling_dict(
                        dataset, sampling_diversity.keys(), batch_size * 10
                    )
                    print(self.sampling_dict)
                    raise
            if filter_speakers is not None:
                if not random_speaker:
                    orig_len = len(dataset.data)
                    filtered_speakers = dataset.data["speaker"].value_counts().sort_values(ascending=False)[:filter_speakers].index.tolist()
                    dataset.data = dataset.data[dataset.data["speaker"].isin(filtered_speakers)]
                    final_len = len(dataset.data)
                    print(f"Filtered dataset down to {filter_speakers} speakers, conserving {final_len/orig_len*100:.2f}% of the data")
            for item in DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset._collate_fn,
                num_workers=multiprocessing.cpu_count(),
            ):
                if total_hours < 10:
                    copy = False
                else:
                    copy = True
                if not copy:
                    speaker_keys = []
                    if not random_speaker:
                        for i in range(len(item["speaker_key"])):
                            if item["speaker_key"][i] in dataset2model:
                                speaker_key = dataset2model[item["speaker_key"][i]]
                            else:
                                speaker_key = item["speaker_key"][i]
                            if speaker_key not in self.model.speaker2dvector.keys():
                                print(
                                    f"WARNING: Speaker {speaker_key} not found in model, random speaker will be used"
                                )
                                speaker_key = list(self.model.speaker2dvector.keys())[
                                    np.random.randint(len(self.model.speaker2dvector))
                                ]
                            speaker_keys.append(speaker_key)
                    else:
                        speaker_keys = random.choices(
                            random_keys,
                            weights=[1/random_weights[k] for k in random_keys],
                            k=len(item["speaker_key"])
                        )
                        for speaker_key in speaker_keys:
                            random_weights[speaker_key] += 1
                    if random_speaker:
                        item["speaker"] = torch.tensor(
                            [random_vecs[x] for x in speaker_keys]
                        ).to(self.device)
                        item["speaker_key"] = speaker_keys
                    else:
                        item["speaker"] = torch.tensor(
                            [self.model.speaker2dvector[x] for x in speaker_keys]
                        ).to(self.device)
                        item["speaker_key"] = speaker_keys
                if not copy:
                    audios = self.generate_samples(
                        item,
                        increase_diversity=increase_diversity,
                        fixed_diversity=fixed_diversity,
                        sampling_diversity=sampling_diversity,
                        oracle_diversity=oracle_diversity,
                        prior_sampling=prior_sampling,
                    )
                else:
                    audios = []
                    speaker_keys = []
                    for i in range(len(item["mel"])):
                        real_mel = item["mel"][i][
                            : torch.sum(item["duration"][i])
                        ].cpu()
                        audios.append(int16_samples_to_float32(self.synth(real_mel)[0]))
                        speaker_keys.append(item["speaker_key"][i])
                        #if self.augmentations is not None:
                        #    audios = [self.augmentations(m, sample_rate=self.model.hparams.sampling_rate) for m in audios]
                for i, audio in enumerate(audios):
                    save_dir = Path(target_dir, Path(speaker_keys[i]).name)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(
                        save_dir / Path(item["id"][i]).with_suffix(".wav"),
                        torch.tensor(audio).unsqueeze(0),
                        sample_rate=22050,
                    )
                    open(save_dir / Path(item["id"][i]).with_suffix(".lab"), "w").write(
                        item["text"][i]
                    )
                    add_hours = len(audio) / self.model.hparams.sampling_rate / 3600
                    pbar.update(add_hours)
                    total_hours += add_hours
                    if total_hours > hours:
                        break
                if total_hours > hours:
                    break
