from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

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
    #ds = TTSDataset(
    #    LibrittsDataset(target_directory="../Data/LibriTTS/train-clean-100-aligned", chunk_size=10_000),
    #    priors=[],
    #    variances=[],
    #    variance_transforms=["none", "none", "none"],
    #    denoise=False,
    #)
    #ds = TTSDataset(
    #    LibrittsDataset(target_directory="../Data/LibriTTS/train-clean-360-aligned", chunk_size=10_000),
    #    priors=[],
    #    variances=[],
    #    variance_transforms=["none", "none", "none"],
    #    denoise=False,
    #)
    ds = TTSDataset(
        LibrittsDataset(target_directory="../Data/LibriTTS/dev-clean-aligned", chunk_size=10_000),
        priors=["pitch", "energy", "snr", "duration"],
        variances=["pitch", "energy", "snr"],
        variance_transforms=["none", "none", "none"],
        variance_levels=["phone","phone", "phone"],
        denoise=False,
        overwrite_stats=True,
    )
    min_idx = -1
    min_len = float("inf")
    for i, item in tqdm(enumerate(ds), total=10):
        if item["duration"].sum() < min_len:
            min_len = item["duration"].sum()
            min_idx = i
        if i > 10:
            break
    print(min_idx)
    fig = ds.plot(min_idx, show=False)
    fig.save("test.png")
    # print(train_ud[3398])
    # train_ud.plot(3398)
    # train_ud.plot(3692)
