import torch
import numpy as np
from librosa.filters import mel as librosa_mel
from librosa.util import pad_center, tiny
from scipy.signal import get_window
import torch.nn.functional as F

# TODO: greaty simplify this by porting it to nnAudio or similar

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
    normal_indices = np.logical_and(new_values_abs > lower, new_values_abs < upper, new_values_abs != 0)
    normal_indices = np.array([True] + list(normal_indices))
    new_values = np.array(values)
    new_values[~normal_indices] = np.nan
    new_values[new_values == 0] = np.nan
    nans, x = nan_helper(new_values)
    try:
        new_values[nans] = np.interp(x(nans), x(~nans), new_values[~nans])
    except ValueError:
        print(new_values)
        return None
    return new_values

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            torch.autograd.Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

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
        self.stft_fn = STFT(filter_length, hop_length, win_length).to("cpu")
        mel_basis = librosa_mel(sr=sampling_rate, n_fft=filter_length,n_mels=n_mel,fmin=mel_fmin,fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def forward(self, y):
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
    sil_phones = ["sil", "sp", "spn", ""]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    counter = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # add silence phone if timestamp gap occurs
        if s != end_time and len(phones) > 0:
            phones.append("sil")
            durations.append(
                int(
                    np.round(s * sampling_rate / hop_length)
                    - np.round(end_time * sampling_rate / hop_length)
                )
            )

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
            phones.append("sil")
            end_time = e

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
