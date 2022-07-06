CUDA_VISIBLE_DEVICES="0" pdm run python src/train.py \
--accelerator gpu \
--batch_size 12 \
--accumulate_grad_batches 4 \
--precision 16 \
--max_epochs 30