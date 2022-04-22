python finetune.py \
--do_train \
--gpu_id 2 \
--maxlen 128 \
--batch_size 64 \
--struc 'cls' \
--warmup \
--epoch 5 \
--model_type 'nezha_base' \
--use_avg \
--use_fgm


