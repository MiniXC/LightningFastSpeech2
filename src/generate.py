from alignments.datasets.libritts import LibrittsDataset
import torchaudio
import torch

from synthesis.generator import SpeechGenerator
from synthesis.g2p import EnglishG2P
from dataset.datasets import TTSDataset

if __name__ == "__main__":
    orig_ds = TTSDataset(
        LibrittsDataset(target_directory="../data/train-clean-aligned"),
        priors=[],
    )
    transfer_ds = TTSDataset(
        LibrittsDataset(target_directory="../data/dev-clean-aligned"),
        priors=[],
    )
    generator = SpeechGenerator("models/early_stop_js_frame.ckpt", EnglishG2P(), device="cuda:2", overwrite=True)
    generator.generate_from_dataset(transfer_ds, "../data/synth/first_test", speaker2dvector=orig_ds.speaker2dvector)
    # audio = generator.generate_sample_from_text(
    #     "Hello, this is an important test!",
    #     speaker2dvector=train_ds.speaker2dvector,
    # )
    # torchaudio.save("test.wav", torch.tensor(audio).unsqueeze(0), sample_rate=22050, encoding="PCM_S")
