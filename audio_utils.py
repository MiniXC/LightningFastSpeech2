import torch
import numpy as np
from librosa.filters import mel as librosa_mel


def dynamic_range_compression(x, C=1, clip_val=1e-7):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def remove_outliers(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)
    values[~normal_indices] = 0
    return values


def remove_outliers_new(values):
    for i, p in enumerate(values):
        if (
            p == 0
            and i > 0
            and i + 1 < len(values)
            and values[i - 1] > 0
            and values[i + 1] > 0
        ):
            values[i] = (values[i - 1] + values[i + 1]) / 2
        if (
            p > 0
            and i > 0
            and i + 1 < len(values)
            and values[i - 1] == 0
            and values[i + 1] == 0
        ):
            values[i] = 0
    new_values = np.array(values)
    new_values = new_values[1:] - new_values[:-1]
    new_values_abs = abs(new_values)
    p25 = np.percentile(new_values_abs, 25)
    p75 = np.percentile(new_values_abs, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(new_values_abs > lower, new_values_abs < upper)
    normal_indices = np.array([True] + list(normal_indices))
    new_values = np.array(values)
    new_values[~normal_indices] = np.nan
    new_values[new_values == 0] = np.nan
    nans, x = nan_helper(new_values)
    new_values[nans] = np.interp(x(nans), x(~nans), new_values[~nans])
    return new_values


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel
        self.sampling_rate = sampling_rate
        self.stft_fn = torch.stfn
        mel_basis = librosa_mel(sampling_rate, filter_length, n_mel, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float().to("cuda")
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        energy = torch.norm(magnitudes, dim=1)

        return mel_output, energy


def get_alignment(tier, sampling_rate, hop_length):
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time
