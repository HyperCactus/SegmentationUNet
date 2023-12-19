"""
Coppied from https://www.kaggle.com/code/kashiwaba/sennet-hoa-inference-unet-simple-baseline
"""

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from global_params import *
import albumentations as A
from utils import *
from modules import ImprovedUNet
from dataset import VAL_LOADER, TRAIN_LOADER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tesd_dir = '/kaggle/input/blood-vessel-segmentation/test/'

def main():
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)
    
    kidney_paths = glob(os.path.join(tesd_dir, 'kidney_*'))

    save_predictions(kidney_paths, model, device=device)
    
    prediction_rles = create_rle_df(kidney_paths, subdir_name='preds')
    
    prediction_rles.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()