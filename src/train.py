from argparse import ArgumentParser

from dataset.datasets import UnprocessedDataset
import torch

parser = ArgumentParser()

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    train_ud = UnprocessedDataset(
        "../Data/LibriTTS/train-clean-360-aligned",
        max_entries=10_000,
        pitch_quality=0.25,
    )
    train_ud.plot(1_000)
    # print(train_ud[3398])
    # train_ud.plot(3398)
    # train_ud.plot(3692)
