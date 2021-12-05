import configparser
import warnings

warnings.filterwarnings('ignore')
config = configparser.ConfigParser()
config.read("config.ini")

from datasets import load_dataset
import datasets
import librosa

dataset = load_dataset("multilingual_librispeech_opus.py", "german", split="validation")

def load_audio(x):
    x["audio"], _ = librosa.load(x["file"], sr=22500, res_type="kaiser_fast")
    return x

dataset.map(
    load_audio,
    num_proc=16,
    features=datasets.Features(
        {
            "file": datasets.Value("string"),
            "audio": datasets.features.Sequence(datasets.Value("float32")),
            "text": datasets.Value("string"),
            "speaker_id": datasets.Value("int64"),
            "chapter_id": datasets.Value("int64"),
            "id": datasets.Value("string"),
        }
    )
)