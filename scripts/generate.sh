pdm run python litfass/generate.py \
--checkpoint_path "models/fastdiff_nopretrain.ckpt" \
--dataset "../data/dev-clean-aligned" \
--output_path "../generated/fastdiff_nopretrain" \
--hours .5 \
--batch_size 1 \
--use_voicefixer True \
--cache_path "../dataset_cache" \
--tts_device "cuda:0" \
--hifigan_device "cuda:1" \
--use_fastdiff True \
--fastdiff_n 4 \
--min_samples_per_speaker 0 \
--num_workers 16