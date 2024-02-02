import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from modules import ImprovedUNet
from glob import glob
from utils import inference_fn, load_checkpoint, plot_examples, remove_small_objects
from dataset import preprocess_image, preprocess_mask
from global_params import *
        
        
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedUNet(in_channels=IN_CHANNELS, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)

    plot_examples(model, num=5, device=device, dataset_folder=VAL_DATASET_DIR)
    
    # img = preprocess_image('data_png/train/kidney_1_dense/images/0610.png')
    # img = preprocess_image('data_png/train/kidney_2/images/1262.png')
    # mask = preprocess_mask('data_png/train/kidney_2/labels/1262.png')
    # c, h, w = img.shape
    # print(f'Original size: {h}x{w}')
    # nh, nw = img.shape[1:]
    # img = img.permute(1, 2, 0).numpy()
    # mask = mask.numpy().astype(np.uint8)
    # print(f'Mask shape: {mask.shape}')
    # mask = remove_small_objects(mask, 600)
    # print(f'Sum of mask: {np.sum(mask)}')
    
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask)
    # plt.show()
    
    
