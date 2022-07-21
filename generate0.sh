pdm run python src/generate.py \
--tts_device "cuda:2" \
--hifigan_device "cuda:3" \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/baseline.ckpt \
--output_path ../Data/synth/synth_ns_25 \
--filter_speakers 25 \
--augment True \

pdm run python src/generate.py \
--tts_device "cuda:2" \
--hifigan_device "cuda:3" \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/baseline.ckpt \
--output_path ../Data/synth/synth_ns_50 \
--filter_speakers 50 \
--augment True \

pdm run python src/generate.py \
--tts_device "cuda:2" \
--hifigan_device "cuda:3" \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/baseline.ckpt \
--output_path ../Data/synth/synth_ns_100 \
--filter_speakers 100 \
--augment True \

pdm run python src/generate.py \
--tts_device "cuda:2" \
--hifigan_device "cuda:3" \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/baseline.ckpt \
--output_path ../Data/synth/synth_ns_200 \
--filter_speakers 200 \
--augment True \

pdm run python src/generate.py \
--tts_device "cuda:2" \
--hifigan_device "cuda:3" \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/baseline.ckpt \
--output_path ../Data/synth/synth_ns_500 \
--filter_speakers 500 \
--augment True \

pdm run python src/generate.py \
--tts_device "cuda:2" \
--hifigan_device "cuda:3" \
--dataset_path ../data/train-clean-aligned \
--checkpoint_path models/baseline.ckpt \
--output_path ../Data/synth/synth_ns_900 \
--filter_speakers 900 \
--augment True \

# pdm run python src/generate.py \
# --tts_device "cuda:2" \
# --hifigan_device "cuda:3" \
# --dataset_path ../data/train-clean-aligned \
# --checkpoint_path models/early_stop_js_phone_full.ckpt \
# --output_path ../Data/synth/snr_oracle \
# --augment True \
# --snr_oracle True

# pdm run python src/generate.py \
# --tts_device "cuda:2" \
# --hifigan_device "cuda:3" \
# --dataset_path ../data/train-clean-aligned \
# --checkpoint_path models/early_stop_js_phone_full.ckpt \
# --output_path ../Data/synth/all_oracle \
# --augment True \
# --snr_oracle True \
# --energy_oracle True \
# --pitch_oracle True \
# --duration_oracle True

# pdm run python src/generate.py \
# --tts_device "cuda:2" \
# --hifigan_device "cuda:3" \
# --dataset_path ../data/train-clean-aligned \
# --checkpoint_path models/early_stop_js_phone_full.ckpt \
# --output_path ../Data/synth/energy_oracle \
# --augment True \
# --energy_oracle True

# pdm run python src/generate.py \
# --tts_device "cuda:2" \
# --hifigan_device "cuda:3" \
# --dataset_path ../data/train-clean-aligned \
# --checkpoint_path models/early_stop_js_phone_full.ckpt \
# --output_path ../Data/synth/pitch_oracle \
# --augment True \
# --pitch_oracle True

# --augment True \
# random factor -> multiply orginal series [0.75, 1.25]
# sample -> sample from original 20% of the time
# outlier_sample -> sample from outliers 10% of the time