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
from dataset import VAL_LOADER, TRAIN_LOADER, UsageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tesd_dir = '/kaggle/input/blood-vessel-segmentation/test/'

def test_transform(image):
    
    image_np = image.permute(1, 2, 0).numpy()

    transform = A.Compose([
        A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
        A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=True),  # High pass filter
    ])

    augmented = transform(image=image_np)
    augmented_image = augmented['image']

    augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)

    return augmented_image

def main():
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)

    k5_path = 'data/test/kidney_5'
    k6_path = 'data/test/kidney_6'
    
    test_dir = 'data/test'

    test_kidney_dirs = glob(test_dir)
    print(test_kidney_dirs)

    for kidney_dir in test_kidney_dirs:
        img_dir = os.path.join(kidney_dir, 'images')
        img_files = sorted([os.path.join(img_dir, f) 
                            for f in os.listdir(img_dir) if f.endswith('.tif')])

        test_set = UsageDataset(img_files, transform=test_transform)

        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        os.makedirs(os.path.join(kidney_dir, 'predictions'), exist_ok=True)

        model.eval()
        with torch.no_grad():
            for i, (image, _) in enumerate(tqdm(test_loader)):
                image = image.to(device=device)
                preds = torch.sigmoid(model(image))
                preds = (preds > PREDICTION_THRESHOLD).float()
                preds = preds.to('cpu').numpy()

                rle = rle_encode(preds)

            
        
        for i, (image, _) in enumerate(tqdm(k6_loader)):
            image = image.to(device=device)
            preds = torch.sigmoid(model(image))
            preds = (preds > PREDICTION_THRESHOLD).float()
            # save_predictions_as_imgs(image, preds, folder='test_predictions/kidney_6', device=device)
    
    # kidney_paths = glob(os.path.join(tesd_dir, 'kidney_*'))

    # save_predictions(kidney_paths, model, device=device)
    
    # prediction_rles = create_rle_df(kidney_paths, subdir_name='preds')
    
    # prediction_rles.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()