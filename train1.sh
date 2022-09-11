CUDA_VISIBLE_DEVICES="2,3" pdm run python src/train.py \
--from_checkpoint models/BFF-v2.ckpt \
--accelerator gpu \
--devices 2 \
--strategy ddp \
--precision 16 \
--batch_size 1 \
--val_check_interval 0.1 \
--accumulate_grad_batches 32 \
--layer_dropout 0.05 \
--max_epochs 30 \
--gradient_clip_val 1.0 \
--encoder_hidden 768 \
--encoder_conv_filter_size 1536 \
--variance_filter_size 768 \
--duration_filter_size 768 \
--decoder_hidden 768 \
--decoder_conv_filter_size 1536 \
--encoder_head 8 \
--decoder_head 8 \
--variance_levels phone phone phone \
--variance_transforms none none none \
--variance_early_stopping js \
--encoder_layers 6 \
--encoder_kernel_sizes 5 19 23 13 10 6 \
--decoder_layers 8 \
--decoder_kernel_sizes 17 20 21 15 9 9 13 15 \
--priors energy duration snr pitch \
--wandb_name "BFF" \
--train_target_path "../data/train-clean-100-aligned" "../data/train-clean-360-aligned" "../data/train-other-500-aligned" \
--train_source_path "../Data/LibriTTS/train-clean-100" "../Data/LibriTTS/train-clean-360" "../Data/LibriTTS/train-other-500" \
--train_source_url "https://www.openslr.org/resources/60/train-clean-100.tar.gz" "https://www.openslr.org/resources/60/train-clean-360.tar.gz" "https://www.openslr.org/resources/60/train-other-500.tar.gz"


# --strategy deepspeed_stage_2 \
# --devices 4 \

# ddp_find_unused_parameters_false