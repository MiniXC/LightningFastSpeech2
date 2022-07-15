pdm run python src/generate.py \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/early_stop_js_phone_full.ckpt \
--output_path ../Data/synth/gen_pitch_25 \
--pitch_diversity 0.25