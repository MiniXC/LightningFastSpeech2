from scipy.signal import cwt, ricker
import numpy as np

# https://www.isca-speech.org/archive_v0/ssw8/papers/ssw8_285.pdf 2.3
# implementation https://github.com/ming024/FastSpeech2/issues/136


def wavelet_decomposition(signal, wavelet, n_scales=10, tau=0.2833425):
    widths = [2 ** (i + 1) * tau for i in range(1, n_scales + 1)]
    cwtmatr = cwt(signal, wavelet, widths)

    constant = [(i + 2.5) ** (-5 / 2) for i in range(1, n_scales + 1)]
    constant = np.array(constant)[:, None]
    cwtmatr = cwtmatr * constant
    return cwtmatr, widths


def wavelet_recomposition(wavelet_matrix):
    signal = wavelet_matrix.sum(axis=0)
    signal = (signal - signal.mean()) / (signal.std() + 1e-7)
    return signal


class CWT:
    def __init__(self, wavelet=ricker, n_scales=10, tau=0.2833425):
        self.wavelet = wavelet
        self.n_scales = n_scales
        self.tau = tau

    def decompose(self, signal):
        signal[signal == 0] = 1e-7
        original_signal = signal.copy()
        signal = np.log(signal)
        cwtmatr, widths = wavelet_decomposition(
            (signal - signal.mean()) / (signal.std() + 1e-7),
            self.wavelet,
            self.n_scales,
            self.tau,
        )
        return {
            "signal": signal,
            "original_signal": original_signal,
            "spectrogram": cwtmatr.T,
            "mean": signal.mean(),
            "std": signal.std(),
        }

    def recompose(self, spectrogram, mean, std):
        signal = wavelet_recomposition(spectrogram)
        return signal * std + mean
