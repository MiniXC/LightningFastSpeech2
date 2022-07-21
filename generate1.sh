pdm run python src/generate.py \
--tts_device "cuda:2" \
--hifigan_device "cuda:3" \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/baseline.ckpt \
--output_path ../Data/synth/snr_baseline \
--augment True

pdm run python src/generate.py \
--tts_device "cuda:2" \
--hifigan_device "cuda:3" \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/no_snr-v1.ckpt \
--output_path ../Data/synth/baseline \
--augment True

# --augment True \
# random factor -> multiply orginal series [0.75, 1.25]
# sample -> sample from original 20% of the time
# outlier_sample -> sample from outliers 10% of the time