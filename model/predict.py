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





def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    if rle=='':
        rle = '1 0'
    return rle


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)
    model.eval()
    
    
    rles = []
    pbar = tqdm(enumerate(VAL_LOADER), total=len(VAL_LOADER), desc='Inference ')
    for step, (images, shapes) in pbar:
        shapes = shapes.numpy()
        images = images.to(device, dtype=torch.float)
        with torch.no_grad():
            preds = model(images)
            preds = (nn.Sigmoid()(preds)>0.5).double()
        preds = preds.cpu().numpy().astype(np.uint8)

        for pred, shape in zip(preds, shapes):
            pred = cv2.resize(pred[0], (shape[1], shape[0]), cv2.INTER_NEAREST)
            # pred = remove_small_objects(pred, 10)
            rle = rle_encode(pred)
            rles.append(rle)


    ids = []
    for p_img in tqdm(ls_images):
        path_ = p_img.split(os.path.sep)
        # parse the submission ID
        dataset = path_[-3]
        slice_id, _ = os.path.splitext(path_[-1])
        ids.append(f"{dataset}_{slice_id}")

    submission = pd.DataFrame.from_dict({
        "id": ids,
        "rle": rles
    })
    submission.to_csv("submission.csv", index=False)