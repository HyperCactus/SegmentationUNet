import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from modules import ImprovedUNet
from glob import glob
from utils import inference_fn, load_checkpoint, plot_examples, min_size
from dataset import preprocess_image, preprocess_mask
from global_params import *
        
        
if __name__ == '__main__':
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model = ImprovedUNet(in_channels=IN_CHANNELS, out_channels=1).to(device=device)
    # load_checkpoint(torch.load(CHECKPOINT_DIR), model)

    # plot_examples(model, num=5, device='cpu', dataset_folder=VAL_DATASET_DIR, sub_data_idxs=(500, 1400))
    
    # img = preprocess_image('data_png/train/kidney_1_dense/images/0610.png')
    img = preprocess_image('data_png/train/kidney_3_dense/images/0610.png')
    c, h, w = img.shape
    print(f'Original size: {h}x{w}')
    img = min_size(img)
    nh, nw = img.shape[1:]
    print(f'New size: {nh}x{nw}')
    img = img.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.show()
