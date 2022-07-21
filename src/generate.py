from argparse import ArgumentParser
import pickle
from pathlib import Path

from alignments.datasets.libritts import LibrittsDataset
import torchaudio
import torch
from audiomentations import Compose, PitchShift, RoomSimulator, AddGaussianSNR

from synthesis.generator import SpeechGenerator
from synthesis.g2p import EnglishG2P
from dataset.datasets import TTSDataset

from third_party.argutils import str2bool

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

    parser.add_argument("--pitch_fixed", type=float, default=None)
    parser.add_argument("--energy_fixed", type=float, default=None)
    parser.add_argument("--snr_fixed", type=float, default=None)
    parser.add_argument("--duration_fixed", type=float, default=None)

    # oracles
    parser.add_argument("--pitch_oracle", type=str2bool, default=False)
    parser.add_argument("--energy_oracle", type=str2bool, default=False)
    parser.add_argument("--snr_oracle", type=str2bool, default=False)
    parser.add_argument("--duration_oracle", type=str2bool, default=False)

    # sampling for variance diversity
    parser.add_argument("--pitch_sampling", type=float, default=0.0)
    parser.add_argument("--energy_sampling", type=float, default=0.0)
    parser.add_argument("--snr_sampling", type=float, default=0.0)
    parser.add_argument("--duration_sampling", type=float, default=0.0)

    # sampling level
    parser.add_argument("--sampling_level", type=str, default="all") # can be "all", "phone", "speaker", "phone+speaker"

    # sampling path
    parser.add_argument("--sampling_path", type=str, default="sampling_values")

    parser.add_argument("--augment", type=str2bool, default=False)
    parser.add_argument("--copy", type=str2bool, default=False)
    
    # speakers
    parser.add_argument("--filter_speakers", type=int, default=None)
    parser.add_argument("--random_speaker", type=str2bool, default=False)
    parser.add_argument("--include_dataset_speakers", type=str2bool, default=False)

    args = parser.parse_args()

    args_dict = vars(args)

    if args.augment:
        augmentations = Compose([
            PitchShift(min_semitones=-4, max_semitones=4, p=0.25),
            AddGaussianSNR(min_snr_in_db=10, max_snr_in_db=30.0, p=0.25),
            RoomSimulator(use_ray_tracing=False, p=0.25),
        ])
    else:
        augmentations = None

    diversity_dict = {}
    for key, val in args_dict.items():
        if "diversity" in key and val > 0:
            if "duration" in key:
                diversity_dict["duration"] = val
            else:
                diversity_dict["variances_" + key.split("_")[0]] = val

    fixed_dict = {}
    for key, val in args_dict.items():
        if "fixed" in key and val is not None:
            if "duration" in key:
                fixed_dict["duration"] = val
            else:
                fixed_dict["variances_" + key.split("_")[0]] = val

    sampling_dict = {}
    for key, val in args_dict.items():
        if "sampling" in key and "path" not in key and "level" not in key and val > 0:
            if "duration" in key:
                sampling_dict["duration"] = val
            else:
                sampling_dict["variances_" + key.split("_")[0]] = val

    oracle_dict = {}
    for key, val in args_dict.items():
        if "oracle" in key and val:
            if "duration" in key:
                oracle_dict["duration"] = val
            else:
                oracle_dict["variances_" + key.split("_")[0]] = val

    if Path("ds.pkl").exists() and args.dataset_path == "../data/train-clean-aligned":
        with open("ds.pkl", "rb") as f:
            ds = pickle.load(f)
    else:
        ds = TTSDataset(
            LibrittsDataset(target_directory=args.dataset_path, chunk_size=10_000),
            priors=[],
            variance_transforms=["none", "none", "none"],
        )
        if args.dataset_path == "../data/train-clean-aligned":
            with open("ds.pkl", "wb") as f:
                pickle.dump(ds, f)
    generator = SpeechGenerator(
        args.checkpoint_path,
        EnglishG2P(),
        device=args.tts_device,
        synth_device=args.hifigan_device,
        overwrite=True,
        sampling_path=args.sampling_path,
        augmentations=augmentations,
    )
    generator.generate_from_dataset(
        ds,
        args.output_path,
        batch_size=args.batch_size,
        increase_diversity=diversity_dict,
        fixed_diversity=fixed_dict,
        sampling_diversity=sampling_dict,
        oracle_diversity=oracle_dict,
        filter_speakers=args.filter_speakers,
        copy=args.copy,
        random_speaker=args.random_speaker,
        include_dataset_speakers=args.include_dataset_speakers,
    )
