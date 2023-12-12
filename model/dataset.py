"""
Code copied from https://www.kaggle.com/code/kashiwaba/sennet-hoa-inference-unet-simple-baseline#Prepare
pytorch dataset for the challenge.
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
import albumentations as A
# import segmentation_models_pytorch as smp




def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def load_msk(path):
    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk = msk.astype('float32')
    msk/=255.0
    return msk


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, msk_paths=[], transforms=None):
        self.img_paths  = img_paths
        self.msk_paths  = msk_paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = load_img(img_path)
        
        if len(self.msk_paths)>0:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            orig_size = img.shape
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), torch.tensor(np.array([orig_size[0], orig_size[1]]))


# Usage example

DATASET_FOLDER = "../data"
ls_images = glob(os.path.join(DATASET_FOLDER, "test", "*", "*", "*.tif"))
print(f"found images: {len(ls_images)}")

test_dataset = BuildDataset(ls_images, [])
test_loader = DataLoader(test_dataset, num_workers=0, shuffle=False, pin_memory=True)
# test_dataset = BuildDataset(ls_images, [], transforms=CFG.data_transforms['valid'])
# test_loader = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)