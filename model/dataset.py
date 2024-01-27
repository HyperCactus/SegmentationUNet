# From lab 3

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import os
import cv2
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from glob import glob
from global_params import *

class CustomDataset(Dataset):
    def __init__(self, images_path, masks_path, 
                 augmentation_transforms=None,
                 img_file_ext=IMG_FILE_EXT, mask_file_ext=MASK_FILE_EXT, sub_data_idxs=None):
        
        images_path = [images_path] if isinstance(images_path, str) else images_path
        masks_path = [masks_path] if isinstance(masks_path, str) else masks_path
        assert len(images_path) == len(masks_path), \
            f'Number of images and masks do not match. Found {len(images_path)} images and {len(masks_path)} masks.'
        
        image_files = []
        mask_files = []
        for image_dir, mask_dir in zip(images_path, masks_path):
            image_files += sorted(glob(os.path.join(image_dir, f'*{img_file_ext}')))
            mask_files += sorted(glob(os.path.join(mask_dir, f'*{mask_file_ext}')))
        if sub_data_idxs is not None:
            start = sub_data_idxs[0]
            end = sub_data_idxs[1]
            image_files = image_files[start:end]
            mask_files = mask_files[start:end]
        
        self.image_files = image_files
        self.mask_files = mask_files
        self.augmentation_transforms = augmentation_transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
       
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = min_size(preprocess_image(image_path), min_size=TILE_SIZE)
        mask = min_size(preprocess_mask(mask_path), min_size=TILE_SIZE)

        if self.augmentation_transforms:
            image, mask = self.augmentation_transforms(image, mask)

        return image, mask


class UsageDataset(Dataset):
    def __init__(self, image_files, 
                 input_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
                 augmentation_transforms=None):
        self.image_files = image_files
        self.input_size = input_size
        self.augmentation_transforms = augmentation_transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
       
        image_path = self.image_files[idx]

        image, orig_size = preprocess_image(image_path, return_size=True)
        # orig_size = image.shape

        if self.augmentation_transforms:
            image = self.augmentation_transforms(image)

        return image, torch.tensor(np.array([orig_size[0], orig_size[1]]))


def preprocess_image(path):
    # print(f'path: {path}')
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # print(f'fresh process image img.shape: {img.shape}')
    
    if IN_CHANNELS == 1:
        img = np.tile(img[...,None],[1, 1, 1])
    else:
        img = np.tile(img[...,None],[1, 1, 3]) 
    img = img.astype('float32') 

    # normalize and mean center the image
    mx = np.max(img)
    if mx:
        img/=mx

    orig_size = img.shape
    
    # print(f'process image img.shape: {img.shape}')
    img = np.transpose(img, (2, 0, 1))
    img_ten = torch.tensor(img)
    return img_ten

def preprocess_mask(path):
    
    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk = msk.astype('float32')
    msk/=255.0
    msk = remove_small_objects(msk.astype(np.uint8), 500)
    msk_ten = torch.tensor(msk)
    
    return msk_ten

def augment_image(image, mask):
    
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=(-0.3, 0.1), rotate_limit=180, shift_limit=0.1, p=0.8, border_mode=0), # for 1024 model
        # A.ShiftScaleRotate(scale_limit=(-0.1, 0.4), rotate_limit=180, shift_limit=0.1, p=0.8, border_mode=0),
        A.Affine(shear=(-10, 10), p=0.4), # Untested addition (shear transform)
        A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2)),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.9),
                A.MotionBlur(blur_limit=3, p=0.9),
            ],
            p=0.7,
        ),
        A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, p=0.9),
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
        # A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=True),  # High pass filter
    ])

    augmented = transform(image=image_np, mask=mask_np)
    augmented_image, augmented_mask = augmented['image'], augmented['mask']

    augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)
    augmented_mask = torch.tensor(augmented_mask, dtype=torch.float32)

    return augmented_image, augmented_mask

def val_transform(image, mask):
    
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    transform = A.Compose([
        # A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
        A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=False, p=0),  # High pass filter
        # add the identity transform to do nothing
    ])

    augmented = transform(image=image_np, mask=mask_np)
    augmented_image, augmented_mask = augmented['image'], augmented['mask']

    augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)
    augmented_mask = torch.tensor(augmented_mask, dtype=torch.float32)

    return augmented_image, augmented_mask

def min_size(img, min_size=1024):
    """
    If the image is smaller than the min_size on any dimension, 
    then pad the image to be at least min_size on that dimension.
    """
    h, w = img.shape[-2:] # this works for single image or batch
    x_pad = 0 if w >= min_size else (min_size - w) // 2
    y_pad = 0 if h >= min_size else (min_size - h) // 2
    new_img = nn.ZeroPad2d((x_pad, x_pad, y_pad, y_pad))(img)
    return new_img

def remove_small_objects(img, min_size):
    # Find all connected components (labels)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Create a mask where small objects are removed
    new_img = np.zeros_like(img)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            new_img[labels == label] = 1

    return new_img

def create_loader(image_files, mask_files, batch_size, 
                  transform=None, shuffle=False, sub_data_idxs=None):
    
    dataset = CustomDataset(image_files, mask_files, augmentation_transforms=transform, sub_data_idxs=sub_data_idxs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_test_loader(image_files, batch_size, 
                  augmentations=None, shuffle=False):
    
    dataset = UsageDataset(image_files, augmentation_transforms=augmentations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

VAL_LOADER = create_loader(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, transform=None, shuffle=False)

image_dirs = []
mask_dirs = []
for kidney in TRAIN_DATASETS:
    image_dirs.append(os.path.join(BASE_PATH, kidney, 'images'))
    mask_dirs.append(os.path.join(BASE_PATH, kidney, 'labels'))

TRAIN_LOADER = create_loader(image_dirs, mask_dirs, BATCH_SIZE, 
                            transform=augment_image, shuffle=True)


test_mode = False#True
# print(len(kidney_1_voi_loader))

if test_mode:
    # testing the dataset:
    for batch_idx, (batch_images, batch_masks) in enumerate(TRAIN_LOADER):
        # print(f'BATCH {batch_idx+1}')
        # if batch_idx < 200/BATCH_SIZE:
        #     continue
        if batch_idx > (4):
            break
        print("Batch", batch_idx + 1)
        print("Image batch shape:", batch_images.shape)
        print("Mask batch shape:", batch_masks.shape)
        
        for image, mask in zip(batch_images, batch_masks):
            noise = torch.randn_like(image) * NOISE_MULTIPLIER
            image = image + noise
            image = image.permute((1, 2, 0)).numpy()*255.0;
            print(f'image.shape: {image.shape}')
            # image = image.squeeze(0).numpy()*255.0
            # image = image.numpy()*255.0
            # print(f'image.shape postsqueese: {image.shape}')
            image = image.astype('uint8')
            mask = (mask*255).numpy().astype('uint8')
            print(f'mask sum = {np.sum(mask)}')
            # mask = mask.squeeze(0)
            
            # image_filename = os.path.basename(image_path)
            # mask_filename = os.path.basename(mask_path)
            
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            # plt.title(f"Original Image - {image_filename}")
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            # plt.title(f"Mask Image - {mask_filename}")
            
            plt.tight_layout()
            plt.show()
        # break


