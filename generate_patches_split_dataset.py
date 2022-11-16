from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='../SIDD_Medium_Srgb/Data', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../datasets/denoising/sidd/train',type=str, help='Directory for image patches')
parser.add_argument('--val_dir', default='../datasets/denoising/sidd/train',type=str, help='Directory for validation image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=250, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=10, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
val = args.val_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores

noisy_patchDir = os.path.join(tar, 'input')
clean_patchDir = os.path.join(tar, 'groundtruth')
val_noisy_patchDir = os.path.join(val, 'input')
val_clean_patchDir = os.path.join(val, 'groundtruth')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

if os.path.exists(val):
    os.system("rm -r {}".format(val))

os.makedirs(noisy_patchDir, exist_ok=True)
os.makedirs(clean_patchDir, exist_ok=True)
os.makedirs(val_noisy_patchDir,exist_ok=True)
os.makedirs(val_clean_patchDir,exist_ok=True)

#get sorted folders
noisy_files, clean_files = [], []


for instance in os.listdir(path=src):
    files = natsorted(glob(os.path.join(src, instance, '*.jpg')))

    noisy_files_instance = []
    clean_files_instance = []
    for file_ in files:
        filename = os.path.split(file_)[-1]
        if 'GT' in filename:
            clean_files_instance.append(file_)
        else:
            noisy_files_instance.append(file_)
            
    assert len(clean_files_instance) == 1, '{} has {} GT images'.format(instance, len(clean_files_instance))
            
    clean_files_instance = clean_files_instance * len(noisy_files_instance)
    
    clean_files = clean_files + clean_files_instance
    noisy_files = noisy_files + noisy_files_instance
            


def save_files(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]
        if j % 20 == 0:
            cv2.imwrite(os.path.join(val_noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patch)
            cv2.imwrite(os.path.join(val_clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patch)
        else:
            cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patch)
            cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patch)

Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(noisy_files))))