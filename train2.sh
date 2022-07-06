CUDA_VISIBLE_DEVICES="2" pdm run python src/train.py \
--accelerator gpu \
--batch_size 12 \
--accumulate_grad_batches 4 \
--precision 16 \
--max_epochs 30 \
--gradient_clip_val 1.0 \
--encoder_kernel_sizes 9 9 9 9 \
--decoder_kernel_sizes 9 9 9 9