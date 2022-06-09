from dataset.datasets import UnprocessedDataset

if __name__ == "__main__":
    train_ud = UnprocessedDataset(
        "../Data/LibriTTS/train-clean-360-aligned",
        max_entries=10_000,
        pitch_quality=1,
    )
    print(train_ud[100])
    train_ud.plot(100)