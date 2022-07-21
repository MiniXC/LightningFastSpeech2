from argparse import ArgumentParser
import pickle

from tqdm.auto import tqdm
from alignments.datasets.libritts import LibrittsDataset
from dataset.datasets import TTSDataset

parser = ArgumentParser()

if __name__ == "__main__":
    # train_ud = UnprocessedDataset(
    #     "../Data/LibriTTS/train-clean-100-aligned",
    #     max_entries=10_000,
    #     pitch_quality=0.25,
    # )
    # train_ud.plot(1_000)
    ds = TTSDataset(
        LibrittsDataset(target_directory="../Data/LibriTTS/dev-clean-aligned", chunk_size=10_000),
        priors=[],
        variances=[],
        variance_transforms=["none", "none", "none"],
    )
    min_idx = -1
    min_len = float("inf")
    for i, item in tqdm(enumerate(ds), total=100):
        if item["duration"].sum() < min_len:
            min_len = item["duration"].sum()
            min_idx = i
        if i > 100:
            break
    print(min_idx)
    ds.plot(min_idx)
    # print(train_ud[3398])
    # train_ud.plot(3398)
    # train_ud.plot(3692)
