CUDA_VISIBLE_DEVICES="3" pdm run python src/train.py \
--accelerator gpu \
--batch_size 6 \
--accumulate_grad_batches 8 \
--precision 16 \
--max_epochs 30 \
--decoder_layers 6 \
--gradient_clip_val 1.0 \
--decoder_kernel_sizes 17 21 9 9 9 13