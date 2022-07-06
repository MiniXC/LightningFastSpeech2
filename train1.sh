CUDA_VISIBLE_DEVICES="1" pdm run python src/train.py \
--accelerator gpu \
--batch_size 6 \
--accumulate_grad_batches 8 \
--precision 16 \
--max_epochs 30 \
--decoder_depthwise_conv False \
--encoder_depthwise_conv False \
--duration_depthwise_conv False \
--variance_depthwise_conv False