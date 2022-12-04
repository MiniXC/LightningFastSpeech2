#!/usr/bin/bash

CUDA_VISIBLE_DEVICES="0" pdm run python litfass/train.py \
--accelerator gpu \
--precision 16 \
--batch_size 4 \
--accumulate_grad_batches 12 \
--val_check_interval 1.0 \
--log_every_n_steps 10 \
--layer_dropout 0.00 \
--duration_dropout 0.1 \
--variance_dropout 0.1 0.1 0.1 0.1 \
--soft_dtw_gamma 0.01 \
--max_epochs 40 \
--gradient_clip_val 1.0 \
--encoder_hidden 256 \
--encoder_conv_filter_size 1024 \
--variance_filter_size 256 \
--duration_filter_size 256 \
--decoder_hidden 256 \
--decoder_conv_filter_size 1024 \
--encoder_head 2 \
--decoder_head 2 \
--variance_loss_weights 1 1 1 1 \
--duration_loss_weight 1 \
--duration_nlayers 5 \
--variances pitch energy snr srmr \
--variance_levels frame frame frame frame \
--variance_transforms none none none none \
--variance_losses mse mse mse mse \
--variance_early_stopping none \
--early_stopping False \
--decoder_layers 6 \
--decoder_kernel_sizes 9 9 9 9 9 9 \
--speaker_embedding_every_layer False \
--prior_embedding_every_layer False \
--wandb_name "fastdiff_nopretrain_variances_fixed" \
--wandb_mode "offline" \
--speaker_type "dvector" \
--train_target_path "../data/train-clean-a" \
--train_min_samples_per_speaker 50 \
--priors_gmm True \
--priors_gmm_max_components 2 \
--dvector_gmm False \
--priors energy duration snr pitch srmr \
--sort_data_by_length True \
--fastdiff_vocoder True \
--fastdiff_schedule 1 1 \
--fastdiff_variances True #\
#--from_checkpoint "models/fastdiff_nopretrain_variances_fixed-v1.ckpt"

#--fastdiff_vocoder_checkpoint "fastdiff_model/model_ckpt_steps_1000000.ckpt" \
#--from_checkpoint "models/fastdiff_fixed_inf-v2.ckpt"

# --priors energy duration snr pitch \
# --train_target_path "../data/train-clean-100-aligned" "../data/train-clean-360-aligned" "../data/train-other-500-aligned" \

# --devices 4 \
# --strategy "ddp" \

# --valid_example_directory "examples"