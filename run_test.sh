# python3 test.py --arch Uformer --batch_size 1 --gpu '0,1' \
#     --input_dir dataset --result_dir results \
#     --weights weights/uformer32.pth --embed_dim 32 

python3 test.py --arch Uformer --batch_size 1 --gpu '0,1' \
    --input_dir changed_dataset --result_dir results-2 \
    --weights weights/uformer_0730.pth --embed_dim 32 --save_images