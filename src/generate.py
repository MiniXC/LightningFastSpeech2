from argparse import ArgumentParser

from alignments.datasets.libritts import LibrittsDataset
import torchaudio
import torch

from synthesis.generator import SpeechGenerator
from synthesis.g2p import EnglishG2P
from dataset.datasets import TTSDataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--tts_device", type=str, default="cuda:0")
    parser.add_argument("--hifigan_device", type=str, default="cuda:1")

    # pitch, energy, SNR, duration variance diversity
    parser.add_argument("--pitch_diversity", type=float, default=0.0)
    parser.add_argument("--energy_diversity", type=float, default=0.0)
    parser.add_argument("--snr_diversity", type=float, default=0.0)
    parser.add_argument("--duration_diversity", type=float, default=0.0)

    args = parser.parse_args()

    args_dict = vars(args)

    diversity_dict = {}
    for key, val in args_dict.items():
        if "diversity" in key and val > 0:
            if "duration" in key:
                diversity_dict["duration"] = val
            else:
                diversity_dict["variances_" + key.split("_")[0]] = val

    ds = TTSDataset(
        LibrittsDataset(target_directory=args.dataset_path, chunk_size=10_000),
        priors=[],
        variance_transforms=["none", "none", "none"],
    )
    generator = SpeechGenerator(args.checkpoint_path, EnglishG2P(), device=args.tts_device, synth_device=args.hifigan_device, overwrite=True)
    generator.generate_from_dataset(ds, args.output_path, batch_size=args.batch_size, increase_diversity=diversity_dict)
