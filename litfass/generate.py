from argparse import ArgumentParser
import pickle
from pathlib import Path
import inspect

from alignments.datasets.libritts import LibrittsDataset
from audiomentations import Compose, RoomSimulator, AddGaussianSNR, PitchShift
from huggingface_hub import hf_hub_download
from pytorch_lightning import Trainer

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
    # # load speaker2priors using pickle
    # speaker2priors = pickle.load(open("speaker2priors.pkl", "rb"))
    # model.speaker2priors = speaker2priors
    # model.speaker2dvector = {k.name: v for k, v in model.speaker2dvector.items() if k.name in speaker2priors}
    # model._fit_speaker_prior_gmms()
    # # save checkpoint
    # trainer = Trainer()
    # trainer.model = model
    # trainer.save_checkpoint("lit_model.ckpt")

    generator = SpeechGenerator(
        model,
        EnglishG2P(),
        device=args.tts_device,
        synth_device=args.hifigan_device,
        augmentations=augmentations,
        voicefixer=args.use_voicefixer,
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
        ds = TTSDataset(
            LibrittsDataset(target_directory=args.dataset_path, chunk_size=10_000),
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
        )

        # if args.speaker == "random":
        #     for item in ds:
        #         for random.choice(ds.speakers):
        #             item["speaker"] = speaker
        #             break
