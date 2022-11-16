python3 ./train.py --arch Uformer --batch_size 32 --gpu '0,1' \
    --train_ps 128 --train_dir datasets/train --env 32_0705_1 \
    --val_dir datasets/val --embed_dim 32 --warmup