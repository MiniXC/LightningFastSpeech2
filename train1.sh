CUDA_VISIBLE_DEVICES="0" pdm run python src/train.py \
--accelerator gpu \
--devices 1 \
--precision 16 \
--batch_size 6 \
--val_check_interval 1.0 \
--accumulate_grad_batches 8 \
--layer_dropout 0.00 \
--duration_dropout 0.5 \
--variance_dropout 0.5 0.5 0.5 \
--max_epochs 30 \
--gradient_clip_val 1.0 \
--encoder_hidden 256 \
--encoder_conv_filter_size 1024 \
--variance_filter_size 256 \
--duration_filter_size 256 \
--decoder_hidden 256 \
--decoder_conv_filter_size 1024 \
--encoder_head 2 \
--decoder_head 2 \
--variance_loss_weights 1 0.1 0.1 \
--duration_loss_weight 0.5 \
--variance_levels phone phone phone \
--variance_transforms none none none \
--variance_early_stopping js \
--decoder_layers 6 \
--decoder_kernel_sizes 9 9 9 9 9 9 \
--priors energy duration snr pitch \
--speaker_embedding_every_layer False \
--prior_embedding_every_layer False \
--log_every_n_steps 6 \
--wandb_name "a_split" \
--train_target_path "../data/train-clean-a" \


# --from_checkpoint models/BFF-v6.ckpt \
# --strategy ddp \
# --lr 6e-4 \
# --strategy deepspeed_stage_2 \
# --devices 4 \

# ddp_find_unused_parameters_false
