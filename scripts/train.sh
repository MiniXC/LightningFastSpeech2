#!/usr/bin/bash

CUDA_VISIBLE_DEVICES="0" pdm run python litfass/train.py \
--accelerator gpu \
--devices 1 \
--precision 16 \
--batch_size 6 \
--accumulate_grad_batches 8 \
--val_check_interval 1.0 \
--layer_dropout 0.00 \
--duration_dropout 0.5 \
--variance_dropout 0.5 0.5 0.5 0.5 \
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
--variance_loss_weights 0.0001 0.0001 0.0001 0.0001 \
--duration_loss_weight 0.01 \
--duration_nlayers 5 \
--variances pitch energy snr srmr \
--variance_levels frame frame frame frame \
--variance_transforms none none none none \
--variance_losses soft_dtw soft_dtw soft_dtw soft_dtw \
--variance_early_stopping none \
--early_stopping_patience 5 \
--decoder_layers 6 \
--decoder_kernel_sizes 9 9 9 9 9 9 \
--speaker_embedding_every_layer False \
--prior_embedding_every_layer False \
--wandb_name "icassp_environment" \
--wandb_mode "offline" \
--train_target_path "../data/train-clean-a" \
--speaker_type "dvector_utterance" \
--train_min_samples_per_speaker 50 \
--priors_gmm True \
--priors_gmm_max_components 2 \
--dvector_gmm True

# --priors energy duration snr pitch \