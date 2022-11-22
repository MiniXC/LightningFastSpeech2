from argparse import ArgumentParser
import pickle
from pathlib import Path
import inspect

from alignments.datasets.libritts import LibrittsDataset
from audiomentations import Compose, RoomSimulator, AddGaussianSNR, PitchShift
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import json
import hashlib
import numpy as np

from litfass.fastspeech2.fastspeech2 import FastSpeech2
from litfass.synthesis.generator import SpeechGenerator
from litfass.synthesis.g2p import EnglishG2P
from litfass.dataset.datasets import TTSDataset
from litfass.third_party.argutils import str2bool

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--sentence", type=str, default=None)

    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--hub", type=str, default=None)

    parser.add_argument("--output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tts_device", type=str, default=None)
    parser.add_argument("--hifigan_device", type=str, default=None)
    parser.add_argument("--use_voicefixer", type=str2bool, default=True)
    parser.add_argument("--use_fastdiff", type=str2bool, default=False)

    parser.add_argument("--cache_path", type=str, default=None)

    # override priors
    parser.add_argument(
            "--prior_values", nargs="+", type=float, default=[-1, -1, -1, -1]
        )

    parser.add_argument("--augment_pitch", type=str2bool, default=False)
    for pitch_arg in inspect.signature(PitchShift).parameters:
        parser.add_argument(f"--pitch_{pitch_arg}", type=float, default=None)
    parser.add_argument("--augment_room", type=str2bool, default=False)
    for room_arg in inspect.signature(RoomSimulator).parameters:
        if room_arg == "use_ray_tracing":
            parser.add_argument(f"--room_{room_arg}", type=str2bool, default=False)
        else:
            parser.add_argument(f"--room_{room_arg}", type=float, default=None)
    parser.add_argument("--augment_noise", type=str2bool, default=False)
    for noise_arg in inspect.signature(AddGaussianSNR).parameters:
        parser.add_argument(f"--noise_{noise_arg}", type=float, default=None)        

    parser.add_argument("--copy", type=str2bool, default=False)

    # speakers
    parser.add_argument("--speaker", type=str, default=None) # can be "random", "dataset" or a speaker name
    parser.add_argument("--min_samples_per_speaker", type=int, default=0)

    # number of hours to generate
    parser.add_argument("--hours", type=float, default=1.0)

    args = parser.parse_args()

    args_dict = vars(args)

    if args.dataset is not None and args.sentence is not None:
        raise ValueError("You can only specify one of --dataset and --sentence!")

    if args.tts_device is None:
        args.tts_device = args.device
    if args.hifigan_device is None:
        args.hifigan_device = args.device

    augment_list = []
    if args.augment_pitch:
        pitch_args = {}
        for pitch_arg in inspect.signature(PitchShift).parameters:
            pitch_args[pitch_arg] = args_dict[f"pitch_{pitch_arg}"]
        augment_list.append(PitchShift(**pitch_args))
    if args.augment_room:
        room_args = {}
        for room_arg in inspect.signature(RoomSimulator).parameters:
            room_args[room_arg] = args_dict[f"room_{room_arg}"]
        augment_list.append(RoomSimulator(**room_args))
    if args.augment_noise:
        noise_args = {}
        for noise_arg in inspect.signature(AddGaussianSNR).parameters:
            noise_args[noise_arg] = args_dict[f"noise_{noise_arg}"]
        augment_list.append(AddGaussianSNR(**noise_args))

    if len(augment_list) > 0:
        augmentations = Compose(
            augment_list
        )
    else:
        augmentations = None # pylint: disable=invalid-name

    if args.hub is not None:
        args.checkpoint_path = hf_hub_download(args.hub, filename="lit_model.ckpt")
        
    if args.checkpoint_path is None:
        raise ValueError("No checkpoint path or hub identifier specified!")

    model = FastSpeech2.load_from_checkpoint(args.checkpoint_path)

    generator = SpeechGenerator(
        model,
        EnglishG2P(),
        device=args.tts_device,
        synth_device=args.hifigan_device,
        augmentations=augmentations,
        voicefixer=args.use_voicefixer,
        fastdiff=args.use_fastdiff,
    )

    if args.sentence is not None:
        if args.speaker is not None:
            args.speaker = Path(args.speaker)
        audio = generator.generate_from_text(args.sentence, args.speaker, random_seed=0, prior_strategy="gmm", prior_values=args.prior_values)
        if args.output_path is None:
            raise ValueError("No output path specified!")
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        generator.save_audio(audio, Path(args.output_path) / f"{args.sentence.replace(' ', '_').lower()}.wav")

    if args.dataset is not None:
        ds = None
        if args.cache_path is not None:
            cache_path = Path(args.cache_path)
            tts_kwargs = {
                "speaker_type":model.hparams.speaker_type,
                "min_length":model.hparams.min_length,
                "max_length":model.hparams.max_length,
                "variances":model.hparams.variances,
                "variance_transforms":model.hparams.variance_transforms,
                "variance_levels":model.hparams.variance_levels,
                "priors":model.hparams.priors,
                "n_mels":model.hparams.n_mels,
                "n_fft":model.hparams.n_fft,
                "win_length":model.hparams.win_length,
                "hop_length":model.hparams.hop_length,
                "min_samples_per_speaker":args.min_samples_per_speaker,
                "_stats": model.stats,
            }
            hash_kwargs = tts_kwargs.copy()
            hash_kwargs["dataset"] = args.dataset
            ds_hash = hashlib.md5(
                    json.dumps(hash_kwargs, sort_keys=True).encode("utf-8")
            ).hexdigest()
            cache_path = cache_path / (ds_hash + ".pt")
            if cache_path.exists():
                print("Loading from cache...")
                with cache_path.open("rb") as f:
                    ds = pickle.load(f)
        if ds is None:
            ds = TTSDataset(
                LibrittsDataset(target_directory=args.dataset, chunk_size=10_000),
                speaker_type=model.hparams.speaker_type,
                min_length=model.hparams.min_length,
                max_length=model.hparams.max_length,
                variances=model.hparams.variances,
                variance_transforms=model.hparams.variance_transforms,
                variance_levels=model.hparams.variance_levels,
                priors=model.hparams.priors,
                n_mels=model.hparams.n_mels,
                n_fft=model.hparams.n_fft,
                win_length=model.hparams.win_length,
                hop_length=model.hparams.hop_length,
                min_samples_per_speaker=args.min_samples_per_speaker,
                _stats=model.stats,
            )
            if args.cache_path is not None and not cache_path.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as f:
                    pickle.dump(ds, f)

        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=4,
            collate_fn=ds._collate_fn,
            shuffle=False,
        )

        with tqdm(total=args.hours) as pbar:
            for batch in dl:
                skip_speaker = False
                for i, speaker in enumerate(batch["speaker_path"]):
                    speaker_dvec = Path(str(speaker).replace("-b", "-a"))
                    speaker = speaker.name
                    if speaker_dvec not in model.speaker2dvector:
                        skip_speaker = True
                        print(f"The speaker {speaker} is not present in the d-vector collection!")
                        break
                    # if hasattr(model, "speaker_gmms"):
                    #     if speaker not in model.speaker_gmms:
                    #         skip_speaker = True
                    #         print(f"The speaker {speaker} is not present in the GMM collection!")
                    #         break
                    # else:
                    #     model.speaker_gmms = pickle.load(open("speaker_gmms.pkl", "rb"))
                    # p_sample = model.speaker_gmms[speaker].sample()[0][0]
                    # for h, p in enumerate(model.hparams.priors):
                    #     batch[f"priors_{p}"][i] = p_sample[h]
                    # if hasattr(model, "dvector_gmms"):
                    #     dvec = model.dvector_gmms[speaker_dvec].sample()[0][0]
                    #     batch["speaker"][i] = torch.tensor(dvec)
                    # else:
                    # batch["speaker"][i] = torch.tensor(model.speaker2dvector[speaker_dvec]).to(model.device)
                if skip_speaker:
                    continue
                results = generator.generate_samples(
                    batch,
                    return_original=True,
                    return_duration=True,
                )
                i = 0
                stop_loop = False
                for audio, speaker, id in zip(results["audios"], batch["speaker_key"], batch["id"]):
                    if args.output_path is None:
                        raise ValueError("No output path specified!")
                    output_path = Path(args.output_path) / speaker
                    output_path.mkdir(parents=True, exist_ok=True)
                    generator.save_audio(audio, output_path / id)
                    id_name = id.replace(".wav", "")
                    generator.save_audio(
                        results["original_audios"][i], 
                        output_path / f"{id_name}_original.wav", 
                        fs=model.hparams.sampling_rate,
                    )
                    with open(output_path / f"{id_name}.meta", "wb") as f:
                        pickle.dump({"phones": batch["phones"], "durations": results["durations"]}, f)
                    with open(output_path / f"{id_name}.lab", "w", encoding="utf-8") as f:
                        f.write(batch["text"][i])
                    pbar.update(audio.shape[0] / results["fs"] / 3600)
                    if pbar.n >= args.hours:
                        stop_loop = True
                        break
                    i += 1
                if stop_loop:
                    break


    