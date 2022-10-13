from abc import ABC, abstractmethod

import numpy as np
import pyworld as pw
from srmrpy import srmr
import torch
from torchaudio import functional as F

from litfass.dataset.snr import SNR

class SpeechMetric(ABC):
    @abstractmethod
    def __init__(self, sample_rate, win_length, hop_length):
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length

    @abstractmethod
    def get_metric(self, audio, silence_mask):
        pass

    def get_metric_value(self, audio, silence_mask):
        return self.get_metric(audio, silence_mask)[~silence_mask].mean()

    @abstractmethod
    def __str__(self):
        pass

    def _interpolate(self, x):
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        nans, y = nan_helper(x)
        x[nans] = np.interp(y(nans), y(~nans), x[~nans])
        return x

class WADA(SpeechMetric):
    def __init__(self, sample_rate, win_length, hop_length):
        super().__init__(sample_rate, win_length, hop_length)
        self.name = "WADA"

    def get_metric(self, audio, silence_mask):
        audio = audio.astype(np.float32)
        snr = SNR(audio, self.sample_rate)
        wada = snr.windowed_wada(
            window=self.win_length,
            stride=self.hop_length / self.win_length,
            use_samples=True,
        )
        if len(silence_mask) < len(wada):
            wada = wada[:len(silence_mask)]
        wada[silence_mask] = np.nan
        if all(np.isnan(wada)):
            wada = np.zeros_like(wada)
        else:
            wada = self._interpolate(wada)
        return wada

    def get_metric_value(self, audio, silence_mask):
        audio = audio.astype(np.float32, silence_mask)
        audio = audio[~silence_mask]
        snr = SNR(audio, self.sample_rate)
        snr = SNR(audio.astype(np.float32), self.sample_rate)
        return snr.wada()

    def __str__(self):
        return self.name

class Pitch(SpeechMetric):
    def __init__(self, sample_rate, win_length, hop_length):
        super().__init__(sample_rate, win_length, hop_length)
        self.name = "Pitch"

    def get_metric(self, audio, silence_mask):
        pitch, t = pw.dio(
            audio.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
            speed=self.dio_speed,
        )
        pitch = pw.stonemask(
            audio.astype(np.float64), pitch, t, self.sampling_rate
        ).astype(np.float32)
        pitch[pitch == 0] = np.nan
        if len(silence_mask) < len(pitch):
            pitch = pitch[:len(silence_mask)]
        pitch[silence_mask] = np.nan
        if np.isnan(pitch).all():
            pitch[:] = 1e-7
        pitch = self._interpolate(pitch)

    def __str__(self):
        return self.name

class Energy(SpeechMetric):
    def __init__(self, sample_rate, win_length, hop_length):
        super().__init__(sample_rate, win_length, hop_length)
        self.name = "Energy"

    def get_metric(self, audio, silence_mask):
        energy = np.array(
            [
                np.sqrt(
                    np.sum(
                        (
                            audio[
                                x * self.hop_length : (x * self.hop_length)
                                + self.win_length
                            ]
                            ** 2
                        )
                    )
                    / self.win_length
                )
                for x in range(int(np.ceil(len(audio) / self.hop_length)))
            ]
        )
        if len(silence_mask) < len(energy):
            energy = energy[:len(silence_mask)]

    def __str__(self):
        return self.name

class SRMR(SpeechMetric):
    def __init__(self, sample_rate, win_length, hop_length):
        super().__init__(sample_rate, win_length, hop_length)
        self.name = "SRMR"

    def get_metric(self, audio, silence_mask):
        if self.sample_rate != 16000:
            audio = F.resample(
                torch.from_numpy(audio),
                self.sample_rate,
                16000,
            ).numpy()
            
        srmr_values = np.array(
            [
                srmr(
                    audio[
                        x * self.hop_length : (x * self.hop_length)
                        + self.win_length
                    ], 16000
                )
                for x in range(int(np.ceil(len(audio) / self.hop_length)))
            ]
        )
        if len(silence_mask) < len(srmr_values):
            srmr_values = srmr_values[:len(silence_mask)]

    def get_metric_value(self, audio, silence_mask):
        if self.sample_rate != 16000:
            audio = F.resample(
                torch.from_numpy(audio),
                self.sample_rate,
                16000,
            ).numpy()
        return srmr(audio, 16000)

    def __str__(self):
        return self.name