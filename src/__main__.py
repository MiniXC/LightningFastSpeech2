from dataset.datasets import UnprocessedDataset

if __name__ == "__main__":
    train_ud = UnprocessedDataset(
        "../Data/LibriTTS/train-clean-360-aligned",
        max_entries=100,
        pitch_quality=0.25,
    )
    train_ud.plot(15)
    # print(train_ud[3398])
    #train_ud.plot(3398)
    # train_ud.plot(3692)
