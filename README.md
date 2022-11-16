# Uformer: A General U-Shaped Transformer for Image Restoration (CVPR 2022): Modified for NIA Dataset


Paper: https://arxiv.org/abs/2106.03106


## Modifications of the Original Code
We made a new generate_patches python file that splits the NIA deblurring data into training and test datasets. 

## Package dependencies
The project is built with PyTorch 1.7.1, Python3.7, CUDA10.1. For package dependencies, you can install them by:
```bash
pip install -r requirements.txt
```

## Pretrained model
- Uformer32_denoising_sidd.pth [[Google Drive]](https://drive.google.com/file/d/1dS7Lh46SMbncnwRW9zM5AW3cXrvYkjQU/view?usp=sharing): PSNR 39.77 dB.
- Uformer16_denoising_sidd.pth [[Google Drive]](https://drive.google.com/file/d/1H1TKHw2gcKORC-MwSkBp9g93T4B1jh_b/view?usp=sharing): PSNR 39.65 dB.

## Results from the pretrained model
- Uformer_B: [SIDD](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EtcRYRDGWhBIlQa3EYBp4FYBao7ZZT2dPc5k1Qe-CdPh3A?e=PjBMub) | [DND](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Ekv3A5ic_4RChFa9XXquF_MB8M8tFd7spyHGJi_8obycnA?e=W7xeHe) | [GoPro](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/ElFalK0Qb8NHnyvhkSe1APgB5D0urGRMLnu2nBXJhtzdIw?e=D2XBhS) | [HIDE](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Eh4p1_kZ95xIopXDNyhl-Q0B65xX6C3J_fL-TQDbgvofqQ?e=8766eT) | [RealBlur-J](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EpHFC9FauEpHhJDsFruEmmQBJ4_ZZaMgjaO9SHmB_vocaA?e=3a4b8A) | [RealBlur-R](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Eo2EC8rmkapNu9CxcYLwFpYBD8tX8pvfX_60jJIP8TGgGQ?e=yGbkQ8) | [DPDD](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EvVAI84ZvlNChWsZA6QY4IkBc201zdTAs_g2Ufd5l0FgIQ?e=2DTlah)
- Uformer32: [SIDD](https://drive.google.com/file/d/19lohIfoaxXsWS3DtRtxLh1kl9Dm-ACd-/view?usp=sharing) |  [DND](https://drive.google.com/file/d/1vdg0dp6Rpb623cPsJlXR3YjJu_C-Tap8/view?usp=sharing)


## Training
### Denoising
To train `Uformer32(embed_dim=32)` on SIDD, we use 2 V100 GPUs and run for 250 epochs:

```python
python3 ./train.py --arch Uformer --batch_size 32 --gpu '0,1' \
    --train_ps 128 --train_dir ../datasets/denoising/sidd/train --env 32_0705_1 \
    --val_dir ../datasets/denoising/sidd/val --embed_dim 32 --warmup
```

More configuration can be founded in `train.sh`.

## Evaluation
### Denoising

To evaluate `Uformer32` on SIDD, you can run:

```python
python3 ./test.py --arch Uformer --batch_size 1 --gpu '0' \
    --input_dir ../datasets/denoising/sidd/val --result_dir YOUR_RESULT_DIR \
    --weights YOUR_PRETRAINED_MODEL_PATH --embed_dim 32 
```


## Computational Cost

We provide a simple script to calculate the flops by ourselves, a simple script has been added in `model.py`. You can change the configuration and run it via:

```python
python3 model.py
```

> The manual calculation of GMacs in this repo differs slightly from the main paper, but they do not influence the conclusion. We will correct the paper later.


## Citation
If you find this project useful in your research, please consider citing:

```
@inproceedings{Wang2022Uformer,
	title={Uformer: A General U-Shaped Transformer for Image Restoration},
	author={Wang, Zhendong and Cun, Xiaodong and Bao, Jianmin and Zhou, Wengang and Liu, Jianzhuang and Li, Houqiang},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year={2022}
}
```

## Acknowledgement

This code borrows heavily from [MIRNet](https://github.com/swz30/MIRNet) and [SwinTransformer](https://github.com/microsoft/Swin-Transformer).

