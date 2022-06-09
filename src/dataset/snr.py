from typing import Optional, Tuple, Iterable, Union

import numpy as np
from textgrid import TextGrid
from pathlib import Path

class SNR:
    def __init__(
        self,
        values: Iterable[float],
        rate: int,
        rms_window: Optional[int] = None,
        rms_stride: Optional[float] = 0.5,
        vad: Optional[Union[str,Iterable[Tuple[float, float]]]]=None,
    ):
        """
        Creates a new ``SNR`` object tied to specific audio array (given as ``values``) and sampling rate (as ``rate``).
        RMS window and stride can be set on initialisation using ``rms_window`` and ``rms_stride``, or later using the ``SNR.rms`` endpoint.
        When setting RMS on initalisation, this can be reverted to sample-based using the ``SNR.samples`` endpoint.
        A vad file in ``.json`` or ``.TextGrid`` format or a tuple array of the format ``[[word_start_in_seconds: float, word_duration_in_seconds: float],...]`` can be given as well to enable windowed measures on voiced parts only or to use ``SNR.vad_ratio``.
        """
        self._values = values
        self.rate = rate
        self._rms = rms_window
        self._rms_stride = rms_stride
        if isinstance(vad, str):
            vad = SNR.load_vad(vad)
        self.vad = vad

    def __iter__(self):
        return self._values.__iter__()

    def __getitem__(self, key):
        return SNR(self._values[key], self.rate, self._rms, self._rms_stride, self.vad)

    def __len__(self):
        return len(self._values)

    @property
    def duration(self) -> float:
        """
        The duration in fractional seconds.
        """
        return len(self) / self.rate

    def seconds(self, start: float, end: float) -> 'SNR':
        """
        Returns a new ``SNR`` object which only spans from second ``start`` to second ``end``.
        """
        return self[int(self.rate * start) : int(self.rate * end)]

    def rms(self, rms_window: int=20, rms_stride: float=0.5):
        """
        Returns a new ``SNR`` object with rms-based audio.
        ``rms_window`` and ``rms_stride`` determine the window width and stride as a percentage of the width, respectively.
        """
        return SNR(self._values, self.rate, rms_window, rms_stride, self.vad)

    @staticmethod
    def load_vad(file_path: str) -> Iterable[Tuple[float, float]]:
        """
        Loads voice activity detection intervals for words from either a ``.TextGrid`` or ``.json`` file.
        The resulting tuple array can be added to an ``SNR`` object using ``SNR.add_vad`` or be passed on initialisation of a new SNR class.
        """
        if ".wav" in file_path:
            file_path = file_path.replace(".wav", ".json")
        result = []
        if ".textgrid" in file_path.lower():
            tg = TextGrid.fromFile(file_path)
            for t in tg:
                if t.name == "words":
                    for w in t:
                        if len(w.mark) > 0:
                            result.append([w.minTime, w.maxTime - w.minTime])
        elif ".json" in file_path.lower():
            result = [
                [w["startTime"], w["duration"]]
                for w in json.load(open(file_path))["asrResult"]["words"]
            ]
        return result

    @property
    def samples(self) -> 'SNR':
        """
        Returns a new ``SNR`` object with sample-based audio.
        """
        return SNR(self._values, self.rate, None, self._rms_stride, self.vad)

    @property
    def notebook(self) -> 'SNRNotebook':
        """
        The ``SNRNotebook`` endpoint for Jupyter Notebook visualisation.
        """
        return SNRNotebook(self)

    @classmethod
    def from_file(cls, file_path: str, **kwargs) -> 'SNR':
        """
        Uses ``librosa.load`` to create a ``SNR`` object from an audio file path (``file_path``).
        Keyword arguments for ``SNR`` can be passed as well.
        """
        data, rate = librosa.load(file_path)
        snr = cls(data, rate, **kwargs)
        return snr

    def to_file(self, file_path: str):
        """
        Writes the audio to the given file path.
        """
        Path("/".join(file_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
        sf.write(file_path, self.samples.values, self.rate)

    def add_vad(self, vad: Union[str, Iterable[Tuple[float, float]]]):
        """
        Adds the given VAD intervals to this ``SNR`` object.
        Intervals can be given as ``[[word_start_in_seconds: float, word_duration_in_seconds: float],...]``.
        If ``vad`` is a string, this will fall back on ``SNR.load_vad``.
        """
        if isinstance(vad, str):
            vad = SNR.load_vad(vad)
        self.vad = vad
        return self

    @property
    def values(self):
        """
        Returns the audio values. If set to ``SNR.rms`` these will be rms values, if set to ``SNR.samples``, the original audio array will be returned.
        """
        if self._rms is None:
            return self._values
        else:
            rms_arr = []
            step = int(self.rate * (self._rms / 1000))
            for index in np.arange(0, len(self._values), int(step * self._rms_stride)):
                window_values = self._values[index : index + step]
                rms_arr.append(np.sum(window_values ** 2) / len(window_values))
            return np.array(rms_arr)

    @property
    def power(self) -> float:
        """
        The RMS power in dB.
        """
        return 20 * np.log10(np.sqrt(np.sum(self.values ** 2) / len(self.values)))

    @staticmethod
    def normalize(values: Iterable[float]) -> Iterable[float]:
        """
        Returns a normalized version of the given float array.
        """
        a = np.sqrt(len(values) / np.sum(values ** 2))
        return values * a

    def get_augmented(self, noise: "SNR", snr:int=0):
        """
        Combines this ``SNR`` object with the given ``noise`` object at the desired ``snr`` and returns the new noisy ``SNR`` object.
        """
        # get the most noisy area of the noise
        n, s = noise, self
        ns_diff = len(n) - len(s)
        # edge cases
        if n._rms != s._rms:
            raise ValueError("Noise and signal rms window must match.")
        if n.rate != s.rate:
            raise ValueError("Noise and signal rates must match.")
        if ns_diff < 0:
            raise ValueError("Noise shorter than signal, use longer noise.")
        if ns_diff == 0 and np.allclose(n.values, s.values):
            raise ValueError("Noise identical to signal.")
        rms_l = []
        # look at 100 different evenly spaced points in the audio
        for i in range(100):
            start = ns_diff // 100 * i
            rms_l.append((start, n[start : start + len(s)].power))
        rms_l = np.array(rms_l)
        # take the noise segment with maximum power
        start = int(rms_l[rms_l[:, 1].argmax(), 0])
        n = n[start : start + len(s)]
        # normalize
        std = s.values.std()
        n_audio = SNR.normalize(s._values)
        n_noise = SNR.normalize(n._values)
        # actual SNR computation
        factor = 10 ** (-snr / 20)
        return SNR(
            ((n_audio * std) + (n_noise * std * factor)),
            self.rate,
            self._rms,
            self._rms_stride,
            self.vad,
        )

    def _windowed_measure(self, measure, window, stride, use_vad, use_samples):
        windows = self.get_windows(window, stride, return_slices=True, use_samples=use_samples)
        index_arr = []
        value_arr = []
        for index_slice in windows:
            if use_vad:
                start_in = any(
                    [
                        (v[0] <= index_slice.start / self.rate <= v[0] + v[1])
                        for v in self.vad
                    ]
                )
                stop_in = any(
                    [
                        (v[0] <= index_slice.stop / self.rate <= v[0] + v[1])
                        for v in self.vad
                    ]
                )
                if not (start_in or stop_in):
                    continue
            value_arr.append(getattr(self[index_slice], measure))
            index_arr.append(index_slice)
        return np.array(index_arr), np.array(value_arr)

    def get_windows(self, window: int=100, stride: float=0.5, return_slices: bool=False, use_samples: bool=False):
        """
        Used to get the windowed values of this ``SNR`` object.
        If ``return_slices`` is set to ``True``, slices which can be used to index an SNR object are returned (for example, use ``SNR[SNR.get_windows(return_slices=True)[0]]`` to get the first window).
        If ``return_slices`` is set to ``False``, the values present in each window are returned instead.
        ``window`` and ``stride`` determine the window width and stride as a percentage of the width, respectively.
        """
        index_arr = []
        if use_samples:
            step = window
        else:
            step = int(self.rate * (window / 1000))
        for index in np.arange(0, int(np.ceil(len(self._values) / step) * step), int(step * stride)):
            if index > len(self._values) - 1:
                break
            index_slice = slice(index, min(index + step, len(self._values)))
            if return_slices:
                index_arr.append(index_slice)
            else:
                index_arr.append(self[index_slice])
        if return_slices:
            return np.array(index_arr)
        else:
            return index_arr

    @property
    def wada(self):
        """
        Over the entire audio: Return the wada measure as defined in http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf using open-source code provided here: https://gist.github.com/johnmeade/d8d2c67b87cda95cd253f55c21387e75#file-snr-py-L7
        """
        return _wada(self.values)

    def windowed_wada(self, window, stride=0.5, use_vad=False, use_samples=False):
        """
        ``window`` and ``stride`` determine the window width and stride as a percentage of the width, respectively.
        """
        value_arr = []
        result = self._windowed_measure("wada", window, stride, use_vad, use_samples)
        for i, v in zip(*result):
            if v > -20 and v < 100:
                value_arr.append(v)
            else:
                value_arr.append(np.nan)
        return np.array(value_arr)

    @property
    def r(self):
        """
        Over the entire audio: Returns the r measure defined as the log10 of the ratio of the 95th and 5th percentile after taking the absolute value of each sample or RMS and adding a floor at 10e-10.
        """
        return _r(self.values)

    def windowed_r(self, window, stride=0.5, use_vad=False):
        """
        ``window`` and ``stride`` determine the window width and stride as a percentage of the width, respectively.
        """
        index_arr = []
        value_arr = []
        result = self._windowed_measure("r", window, stride, use_vad)
        for i, v in zip(*result):
            if v > 0:
                index_arr.append(i)
                value_arr.append(v)
        return np.array(index_arr), np.array(value_arr)

    def vad_ratio(self, padding:int=10):
        """
        Over the entire audio: Calculate the ratio of the mean power in voice vs. unvoiced regions. This can be infinity when the power in unvoiced regions is zero.
        ``padding`` (given in milliseconds) can make the voice regions smaller when positive, or larger when negative.
        """
        v_factors = []
        v_powers = []
        s_factors = []
        s_powers = []
        last_i = 0
        for v in self.vad:
            v0 = v[0] - padding / 1000
            v1 = v[1] - padding / 1000
            if v0 - last_i > 0:
                selection = self.seconds(last_i, v0)
                if len(selection.values) > 0:
                    s_factors.append(v0 - last_i)
                    s_powers.append(selection.power)
            selection = self.seconds(v0, v0 + v1)
            if len(selection.values) > 0:
                v_factors.append(v1)
                v_powers.append(selection.power)
            last_i = v0 + v1
        v_factors, s_factors = np.array(v_factors), np.array(s_factors)
        v_powers, s_powers = np.array(v_powers), np.array(s_powers)
        v_factors /= v_factors.sum()
        s_factors /= s_factors.sum()
        s_result = (s_powers * s_factors).sum()
        v_result = (v_powers * v_factors).sum()
        return v_result - s_result

g_vals = np.load(Path(__file__).parent.parent / "data" / "wada_values.npy")

def _wada(wav):
    global g_vals
    # Direct blind estimation of the SNR of a speech signal.
    #
    # Paper on WADA SNR:
    #   http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    #
    # This function was adapted from this matlab code:
    #   https://labrosa.ee.columbia.edu/projects/snreval/#9
    #
    # MIT license, John Meade, 2020
    # init
    eps = 1e-20
    # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
    db_vals = np.arange(-20, 101)
    # peak normalize, get magnitude, clip lower bound
    wav = np.array(wav)
    wav = wav / abs(wav).max()
    abs_wav = abs(wav)
    abs_wav[abs_wav < eps] = eps
    # calcuate statistics
    v1 = max(eps, abs_wav.mean())
    v2 = np.log(abs_wav).mean()
    v3 = np.log(v1) - v2
    # table interpolation
    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    # handle edge cases or interpolate
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
    else:
        wav_snr = db_vals[wav_snr_idx] + (v3 - g_vals[wav_snr_idx]) / (
            g_vals[wav_snr_idx + 1] - g_vals[wav_snr_idx]
        ) * (db_vals[wav_snr_idx + 1] - db_vals[wav_snr_idx])
    # Calculate SNR
    dEng = sum(wav ** 2)
    dFactor = 10 ** (wav_snr / 10)
    dNoiseEng = dEng / (1 + dFactor)  # Noise energy
    dSigEng = dEng * dFactor / (1 + dFactor)  # Signal energy
    snr = 10 * np.log10(dSigEng / dNoiseEng)
    return snr
