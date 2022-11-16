# Uformer16
# python3 ./train.py --arch Uformer --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 16_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 16 --warmup

# Uformer32
# python3 ./train.py --arch Uformer --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

    
# UNet
# python3 ./train.py --arch UNet --batch_size 32 --gpu '0,1' \
#     --train_ps 128 --train_dir /cache/SIDD/train --env 32_0701_1 \
#     --val_dir /cache/SIDD/val --embed_dim 32 --warmup

# mkdir datasets

# python generate_patches_split_dataset.py --src_dir data_0728/jpg --tar_dir datasets/train --val_dir datasets/val

python train.py --arch Uformer --batch_size 128 --gpu '0,1' \
    --train_ps 128 --train_dir datasets/train --env 22_0730_1 \
    --val_dir datasets/val --embed_dim 32 --resume --pretrain_weights weights/uformer_0730.pth