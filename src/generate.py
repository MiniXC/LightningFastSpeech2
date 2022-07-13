from alignments.datasets.libritts import LibrittsDataset
import torchaudio
import torch

from synthesis.generator import SpeechGenerator
from synthesis.g2p import EnglishG2P
from dataset.datasets import TTSDataset

if __name__ == "__main__":
    orig_ds = TTSDataset(
        LibrittsDataset(target_directory="../data/train-clean-aligned", chunk_size=10_000),
        priors=[],
    )
    transfer_ds = TTSDataset(
        LibrittsDataset(target_directory="../data/train-clean-360-aligned", chunk_size=10_000),
        priors=[],
    )
    generator = SpeechGenerator("models/early_stop_mae_phone.ckpt", EnglishG2P(), device="cuda:2", synth_device="cuda:3", overwrite=True)
    generator.generate_from_dataset(transfer_ds, "../Data/synth/early_stop_mae_phone", speaker2dvector=orig_ds.speaker2dvector)
