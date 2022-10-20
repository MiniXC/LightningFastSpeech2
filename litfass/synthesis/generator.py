from pathlib import Path
import random
import shutil
import multiprocessing
import pickle
from copy import deepcopy
import random


import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from voicefixer import VoiceFixer


from litfass.third_party.hifigan import Synthesiser
from litfass.fastspeech2.fastspeech2 import FastSpeech2
from litfass.synthesis.g2p import G2P


def int16_samples_to_float32(y):
    """Convert int16 numpy array of audio samples to float32."""
    if y.dtype != np.int16:
        if y.dtype == np.float32:
            return y
        elif isinstance(y, torch.Tensor):
            return y.numpy()
        else:
            raise ValueError(f"input samples not int16 or float32, but {y.dtype}")
    return y.astype(np.float32) / np.iinfo(np.int16).max


class SpeechGenerator:
    def __init__(
        self,
        model: FastSpeech2,
        g2p_model: G2P,
        device: str = "cuda:0",
        synth_device: str = None,
        overwrite: bool = False,
        voicefixer: bool = True,
        sampling_path: str = None,
        augmentations=None,
        speaker_dict=None,
    ):
        if synth_device is None:
            self.synth = Synthesiser(device=device)
        else:
            self.synth = Synthesiser(device=synth_device)
        self.model = model
        self.model.eval()
        self.g2p = g2p_model
        self.device = device
        self.model.to(self.device)
        self.overwrite = overwrite
        self.sampling_path = sampling_path
        self.augmentations = augmentations
        self.speaker_dict = speaker_dict
        if voicefixer:
            self.voicefixer = VoiceFixer()
        else:
            self.voicefixer = None

    @property
    def speakers(self):
        if self.model.hparams.speaker_type == "dvector":
            return self.model.speaker2dvector.keys()
        elif self.model.hparams.speaker_type == "id":
            return self.model.speaker2id.keys()
        else:
            return None

    def save_audio(self, audio, path, fs=None):
        if fs is None:
            if self.voicefixer:
                sampling_rate = 44100
            else:
                sampling_rate = self.model.hparams.sampling_rate
        else:
            sampling_rate = fs
        # make 2D if mono
        if len(audio.shape) == 1:
            audio = torch.tensor(audio).unsqueeze(0)
        else:
            audio = torch.tensor(audio)
        torchaudio.save(path, audio, sampling_rate)

    def generate_from_text(self, text, speaker=None, random_seed=None, prior_strategy="sample", prior_values=[-1, -1, -1, -1]):
        ids = [
            self.model.phone2id[x] for x in self.g2p(text) if x in self.model.phone2id
        ]
        batch = {}
        speaker_name = None
        if self.model.hparams.speaker_type == "dvector":
            if speaker is None:
                while True:
                    # pylint: disable=invalid-sequence-index
                    speaker = list(self.model.speaker2dvector.keys())[
                        np.random.randint(len(self.model.speaker2dvector))
                    ]
                    # pylint: enable=invalid-sequence-index
                    # TODO: remove this when all models are fixed
                    if len(self.model.hparams.priors) > 0:
                        if isinstance(speaker, Path):
                            speaker_name = speaker.name
                    if speaker_name in self.model.speaker2priors:
                        break
            else:
                speaker_name = speaker.name
            batch["speaker"] = torch.tensor([self.model.speaker2dvector[speaker]]).to(
                self.device
            )
            print("Using speaker", speaker)
        if self.model.hparams.speaker_type == "id":
            batch["speaker"] = torch.tensor([self.model.speaker2id[speaker]]).to(
                self.device
            )
            print("Using speaker", speaker)
        if len(self.model.hparams.priors) > 0:
            if speaker_name is None:
                speaker_name = speaker
            if random_seed is not None:
                np.random.seed(random_seed)
            if prior_strategy == "sample":
                priors = self.model.speaker2priors[speaker_name]
                prior_len = len(priors[self.model.hparams.priors[0]])
                random_index = np.random.randint(prior_len)
                for prior in self.model.hparams.priors:
                    batch[f"priors_{prior}"] = torch.tensor([priors[prior][random_index]]).to(self.device)
                    print(f"Using prior {prior} with value {priors[prior][random_index]:.2f}")
            elif prior_strategy == "gmm":
                gmm = self.model.speaker_gmms[speaker_name]
                values = gmm.sample()[0][0]
                for i, prior in enumerate(self.model.hparams.priors):
                    batch[f"priors_{prior}"] = torch.tensor([values[i]]).to(self.device)
                    print(f"Using prior {prior} with value {values[i]:.2f}")
        batch["phones"] = torch.tensor([ids]).to(self.device)
        for i, prior in enumerate(self.model.hparams.priors):
            if prior_values[i] != -1:
                batch[f"priors_{prior}"] = torch.tensor([prior_values[i]]).to(self.device)
                print(f"Overriding prior {prior} with value {prior_values[i]:.2f}")
        return self.generate_samples(batch)[1][0]

    def generate_samples(
        self,
        batch,
        return_original=False,
        return_duration=False,
    ):
        result = self.model(batch, inference=True)
        fs = self.model.hparams.sampling_rate

        audios = []
        durations = []
        for i in range(len(result["mel"])):
            mel = result["mel"][i][~result["tgt_mask"][i]].cpu()
            durations.append(result["duration_rounded"][i].cpu())
            audios.append(int16_samples_to_float32(self.synth(mel)[0]))

        if self.voicefixer is not None:
            fs_new = None
            fixed_audios = []
            for i, audio in enumerate(audios):
                tmp_dir = Path("/tmp/voicefixer")
                tmp_dir.mkdir(exist_ok=True)
                tmp_hash = str(random.getrandbits(128))
                if fs != 22050:
                    audio = F.resample(torch.tensor(audio), fs, 22050)
                    fs = 22050
                pad_width = int(fs * 0.1)
                audio = np.pad(audio, (pad_width, pad_width), constant_values=(0, 0))
                torchaudio.save(tmp_dir / f"{tmp_hash}.wav", torch.tensor([audio]), fs)
                self.voicefixer.restore(
                    input=tmp_dir / f"{tmp_hash}.wav",
                    output=tmp_dir / f"{tmp_hash}_fixed.wav",
                    cuda=True,
                    mode=1,
                )
                fixed_audio, fs_new = torchaudio.load(tmp_dir / f"{tmp_hash}_fixed.wav")
                # remove padding
                fixed_audio = fixed_audio[0].numpy()[pad_width:-pad_width]
                fixed_audios.append(fixed_audio)

        if self.augmentations is not None:
            audios = [
                self.augmentations(audio, sample_rate=self.model.hparams.sampling_rate)
                for audio in audios
            ]

        if return_original and self.voicefixer is not None:
            result = {
                "original_fs": fs,
                "original_audios": audios,
                "fs": fs_new,
                "audios": fixed_audios,
            }
        elif self.voicefixer is not None:
            result = {
                "fs": fs_new,
                "audios": fixed_audios,
            }
        else:
            result = {
                "fs": fs,
                "audios": audios,
            }

        if return_duration:
            result["durations"] = durations

        return result
