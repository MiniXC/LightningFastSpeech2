pdm run python src/generate.py \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/early_stop_js_phone_full.ckpt \
--output_path ../Data/synth/augmented


# random factor -> multiply orginal series [0.75, 1.25]
# sample -> sample from original 20% of the time
# outlier_sample -> sample from outliers 10% of the time