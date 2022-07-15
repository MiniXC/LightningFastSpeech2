from audioop import mul
from pathlib import Path
import shutil

import torch
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
import multiprocessing

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
    def __init__(self, model_path: str, g2p_model: G2P, device: str = "cuda:0", synth_device: str = None, overwrite: bool = False):
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

    @property
    def speakers(self):
        if self.model.hparams.speaker_type == "dvector":
            return self.model.speaker2dvector.keys()
        elif self.model.hparams.speaker_type == "id":
            return self.model.speaker2id.keys()
        else:
            return None

    def generate_sample_from_text(self, text, speaker=None):
        ids = [self.model.phone2id[x] for x in self.g2p(text) if x in self.model.phone2id]
        batch = {}
        if self.model.hparams.speaker_type == "dvector":
            if speaker is None:
                speaker = list(self.model.speaker2dvector.keys())[np.random.randint(len(self.model.speaker2dvector))]
                print("Using speaker", speaker)
            batch["speaker"] = torch.tensor([self.model.speaker2dvector[speaker]]).to(self.device)
        if self.model.hparams.speaker_type == "id":
            batch["speaker"] = torch.tensor([self.model.speaker2id[speaker]]).to(self.device)
        batch["phones"] = torch.tensor([ids]).to(self.device)
        return self.generate_samples(batch)[0]

    def generate_samples(self, batch, increase_diversity={}):
        result = self.model(batch, inference=True)
        if len(increase_diversity) > 0:
            for key, value in increase_diversity.items():
                batch[key] = result[key].float() * np.random.uniform(1.0-value/2, 1.0+value/2)
                if key == "duration":
                    batch[key] = torch.round(batch[key]).int()
                batch[key] = batch[key].to(self.device)
            result = self.model(batch, inference=False)
        mels = []
        for i in range(len(result["mel"])):
            pred_mel = result["mel"][i][~result["tgt_mask"][i]].cpu()
            mels.append(self.synth(pred_mel)[0])
        return mels

    def generate_from_dataset(self, dataset, target_dir, hours=10, batch_size=6, increase_diversity={}):
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
            print(f"Dataset has {len(dataset_dvectors)} speakers, model has {len(model_dvectors)}")
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
                print("WARNING: There are more speakers in the dataset than in the model, this means that some speakers will be picked randomly")
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
            pbar = tqdm(total=hours, desc="Generating Audio")
            total_hours = 0
            np.random.seed(42)
            for item in DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset._collate_fn,
                num_workers=multiprocessing.cpu_count(),
            ):
                speaker_keys = []
                for i in range(len(item["speaker_key"])):
                    if item["speaker_key"][i] in dataset2model:
                        speaker_key = dataset2model[item["speaker_key"][i]]
                    else:
                        speaker_key = item["speaker_key"][i]
                    if speaker_key not in self.model.speaker2dvector.keys():
                        print(f"WARNING: Speaker {speaker_key} not found in model, random speaker will be used")
                        speaker_key = list(self.model.speaker2dvector.keys())[np.random.randint(len(self.model.speaker2dvector))]
                    speaker_keys.append(speaker_key)
                item["speaker"] = torch.tensor([self.model.speaker2dvector[x] for x in speaker_keys]).to(self.device)
                audios = self.generate_samples(item, increase_diversity=increase_diversity)
                for i, audio in enumerate(audios):
                    save_dir = Path(target_dir, Path(speaker_keys[i]).name)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    audio = int16_samples_to_float32(audio)
                    torchaudio.save(save_dir / Path(item["id"][i]).with_suffix(".wav"), torch.tensor(audio).unsqueeze(0), sample_rate=22050)
                    open(save_dir / Path(item["id"][i]).with_suffix(".lab"), "w").write(item["text"][i])
                    add_hours = len(audio) / self.model.hparams.sampling_rate / 3600
                    pbar.update(add_hours)
                    total_hours += add_hours
                    if total_hours > hours:
                        break
                if total_hours > hours:
                        break