#!/usr/bin/bash
export PT_XLA_DEBUG=1
export XLA_USE_32BIT_LONG=1
# export XLA_GET_TENSORS_OPBYOP=1
# export XLA_SYNC_TENSORS_OPBYOP=1
# export XLA_FLAGS="--xla_dump_to=xla.log"
# export TPU_VISIBLE_DEVICES="1"

# --accumulate_grad_batches 3 \

python3 litfass/train.py \
--accelerator tpu \
--devices 1 \
--precision bf16 \
--batch_size 8 \
--val_check_interval 1.0 \
--log_every_n_steps 250 \
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
--wandb_name "tpu" \
--wandb_mode "offline" \
--speaker_type "dvector" \
--train_target_path "../data/train-clean-100-aligned" \
--train_source_path "../data/train-clean-100" \
--train_source_url "https://www.openslr.org/resources/60/train-clean-100.tar.gz" \
--train_min_samples_per_speaker 50 \
--priors_gmm True \
--priors_gmm_max_components 2 \
--dvector_gmm False \
--priors energy duration snr pitch srmr \
--sort_data_by_length True \
--train_pad_to_multiple_of 16 16 \
--fastdiff_vocoder False \
--num_workers 8 \
--fastdiff_variances True

# --fastdiff_schedule 1 1 \
# --fastdiff_variances True \
# --fastdiff_speakers True \

#--devices 1 \
#--from_checkpoint "models/fastdiff_nopretrain_variances_fixed-v1.ckpt"

#--fastdiff_vocoder_checkpoint "fastdiff_model/model_ckpt_steps_1000000.ckpt" \
#--from_checkpoint "models/fastdiff_fixed_inf-v2.ckpt"

# --priors energy duration snr pitch \
# --train_target_path "../data/train-clean-100-aligned" "../data/train-clean-360-aligned" "../data/train-other-500-aligned" \

# --devices 4 \
# --strategy "ddp" \

# --valid_example_directory "examples"

# tpu_spawn_debug