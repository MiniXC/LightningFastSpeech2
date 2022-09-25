from pathlib import Path
from tqdm.auto import tqdm
import random

a_speaker_counts = {}
b_speaker_counts = {}

a_path = Path("../data/train-clean-a")
b_path = Path("../data/train-clean-b")

extensions = [".lab", ".npy", ".TextGrid"]

file_list = list(Path("../data/train-clean-360-aligned").rglob("*.wav")) + list(Path("../data/train-clean-100-aligned").rglob("*.wav"))
# sort
file_list = sorted(file_list)
# random shuffle with seed
random.Random(42).shuffle(file_list)

for wavfile in tqdm(file_list):
    speaker = wavfile.parent.name
    basename = wavfile.name.replace(".wav", "")

    if speaker not in a_speaker_counts:
        a_speaker_counts[speaker] = 0
    if speaker not in b_speaker_counts:
        b_speaker_counts[speaker] = 0

    if a_speaker_counts[speaker] < b_speaker_counts[speaker]:
        a_speaker_counts[speaker] += 1
        tgt_path = a_path / speaker
    else:
        b_speaker_counts[speaker] += 1
        tgt_path = b_path / speaker

    tgt_path.mkdir(parents=True, exist_ok=True)
    (tgt_path / wavfile.name).symlink_to(wavfile.resolve())
    for ext in extensions:
        (tgt_path / (basename + ext)).symlink_to(wavfile.with_suffix(ext).resolve())