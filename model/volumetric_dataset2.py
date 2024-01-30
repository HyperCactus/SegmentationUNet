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
from tqdm import tqdm
from glob import glob
from global_params import *

class VolumetricDataset(Dataset):
    def __init__(self, images_path, masks_path, volume_size=(TILE_SIZE, TILE_SIZE, TILE_SIZE),
                 transforms_2D=None, transforms_3D=None,
                 img_file_ext=IMG_FILE_EXT, mask_file_ext=MASK_FILE_EXT, sub_data_idxs=None):
        
        self.volume_size = volume_size
        self.msk_ext = mask_file_ext
        self.img_ext = img_file_ext
        self.transforms_2D = transforms_2D
        self.transforms_3D = transforms_3D
        
        self.images_paths = [images_path] if isinstance(images_path, str) else images_path
        self.masks_paths = [masks_path] if isinstance(masks_path, str) else masks_path
        assert len(images_path) == len(masks_path), \
            f'Number of images and masks do not match. Found {len(images_path)} images and {len(masks_path)} masks.'
        
        self.vol_img, self.vol_mask = self._add_volume()
        self.cur_axis = 0
        self.count = 0
    
    def _add_volume(self):
        # randomly select a volume from the dataset
        kidney_idx = np.random.randint(0, len(self.images_paths))
        image_path = self.images_paths[kidney_idx]
        mask_path = self.masks_paths[kidney_idx]
        image, mask = load_random_volume(image_path, mask_path, vol_shape=self.volume_size, 
                                         img_file_ext=self.img_ext, msk_file_ext=self.msk_ext)
        
        if self.transforms_3D is not None:
            image, mask = self.transforms_3D(image, mask)
        
        return image, mask
    
    def _reset(self):
        self.vol_img, self.vol_mask = self._add_volume()
        self.cur_axis = 0
        self.count = 0
    
    def _transform_2D(self, image, mask):
        if self.transforms_2D is not None:
            augmented = self.transforms_2D(image=image.numpy(), mask=mask.numpy())
            image, mask = augmented['image'], augmented['mask']
        return image, mask

    def __len__(self):
        """Returns the number of slices in the current volume. 
        (Not actually the length of the dataset as it is randomized each time)

        Returns:
            Int: The number of slices in the current volume.
        """
        return max(self.volume_size)*3

    def __getitem__(self, idx):
        """Gets a slice from a random volume, slices are taken along the x, y, or z axis.

        Args:
            idx (int): index.
        Returns:
            image (torch.Tensor): A 2D slice of shape (1, H, W).
            mask (torch.Tensor): A 2D slice of shape (1, H, W).
        """
        h, w, d = self.volume_size
        if self.count+1 > max(h, w, d)*3:
            self._reset()
        # if idx > max(h, w, d):
        #     self._reset() # reset the volume if we've gone through all the slices
        if self.cur_axis == 0:
            idx = idx % w
            image = self.vol_img[:, :, idx]
            mask = self.vol_mask[:, :, idx]
        elif self.cur_axis == 1:
            idx = idx % h
            image = self.vol_img[:, idx, :]
            mask = self.vol_mask[:, idx, :]
        else:
            idx = idx % d
            image = self.vol_img[idx, :, :]
            mask = self.vol_mask[idx, :, :]
        self.cur_axis = (self.cur_axis + 1) % 3
        self.count += 1
        
        image, mask = self._transform_2D(image, mask)
        return image, mask

def normalize_image(image):
    image = cv2.equalizeHist(image)
    mx = image.max()
    mn = image.min()
    image = (image - mn) / (mx - mn)
    return image

def load_random_volume(img_path, mask_path, vol_shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE), 
                       img_file_ext=IMG_FILE_EXT, msk_file_ext=MASK_FILE_EXT):
    """Loads a random volume from a series of image files.

    Args:
        path (str or os path like): Path to image slices directory.
        vol_shape (tuple, optional): Desired volume shape. Defaults to (TILE_SIZE, TILE_SIZE, TILE_SIZE).
    Returns:
        vol (torch.Tensor): A random 3D volume of shape (H, W, D), with one channel.
    """
    # Assuming the path contains a series of image files
    # Load the images into a 3D volume
    h, w, d = vol_shape
    img_dirs = sorted(glob(os.path.join(img_path, f'*{img_file_ext}')))
    msk_dirs = sorted(glob(os.path.join(mask_path, f'*{msk_file_ext}')))
    orig_h, orig_w = cv2.imread(img_dirs[0]).shape[:2]
    orig_d = len(img_dirs)
    
    x_start = 0 if orig_w-w <= 0 else np.random.randint(0, orig_w - w)
    y_start = 0 if orig_h-h <= 0 else np.random.randint(0, orig_h - h)
    z_start = 0 if orig_d-d <= 0 else np.random.randint(0, orig_d - d)
    
    x_end = x_start + w if x_start + w < orig_w else orig_w
    y_end = y_start + h if y_start + h < orig_h else orig_h
    z_end = z_start + d if z_start + d < orig_d else orig_d
    
    img_vol = torch.zeros((h, w, d))
    msk_vol = torch.zeros((h, w, d))
    
    loop = tqdm([i+z_start for i in range(z_end-z_start)])
    loop.set_description('Loading volume')
    for i, idx in enumerate(loop):
        img = cv2.imread(img_dirs[idx], cv2.IMREAD_GRAYSCALE) # Load: (H, W)
        msk = cv2.imread(msk_dirs[idx], cv2.IMREAD_GRAYSCALE)
        img = normalize_image(img)
        img = img.astype('float32') # Pre-process
        msk = msk.astype('float32')
        msk /= 255.0
        
        img = img[y_start:y_end, x_start:x_end] # Slice
        msk = msk[y_start:y_end, x_start:x_end]
            
        img_vol[..., i] = torch.tensor(img)
        msk_vol[..., i] = torch.tensor(msk)
        
    return img_vol, msk_vol
    






transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.ShiftScaleRotate(scale_limit=(-0.3, 0.1), rotate_limit=180, shift_limit=0.1, p=0.8, border_mode=0), # for 1024 model
        A.ShiftScaleRotate(scale_limit=(-0.1, 0.1), rotate_limit=45, shift_limit=0.1, p=0.8, border_mode=0),
        A.Affine(shear=(-10, 10), p=0.4), # Untested addition (shear transform)
        A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1, 0.05), contrast_limit=(-0.2, 0.2)),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.9),
                A.MotionBlur(blur_limit=3, p=0.9),
            ],
            p=0.7,
        ),
        # A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, p=0.9),
        A.Resize(TILE_SIZE, TILE_SIZE, interpolation=cv2.INTER_NEAREST),
        # A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=True),  # High pass filter
        ToTensorV2()
    ])


def create_loader(image_files, mask_files, batch_size, 
                  transform_2d=None, transform_3d=None, shuffle=False, tile_size=TILE_SIZE):
    
    dataset = VolumetricDataset(image_files, mask_files, volume_size=(tile_size,tile_size,tile_size), transforms_2D=transform_2d, transforms_3D=transform_3d)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

VAL_LOADER = create_loader(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, shuffle=False)

image_dirs = []
mask_dirs = []
for kidney in TRAIN_DATASETS:
    image_dirs.append(os.path.join(BASE_PATH, kidney, 'images'))
    mask_dirs.append(os.path.join(BASE_PATH, kidney, 'labels'))

TRAIN_LOADER = create_loader(image_dirs, mask_dirs, BATCH_SIZE, 
                            transform_2d=transform, shuffle=True)


test_mode = True
# print(len(kidney_1_voi_loader))

if test_mode:
    # testing the dataset:
    for batch_idx, (batch_images, batch_masks) in enumerate(TRAIN_LOADER):
        print(f"Batch {batch_idx+1} of {len(TRAIN_LOADER)}")
        print("Image batch shape:", batch_images.shape)
        print("Mask batch shape:", batch_masks.shape)
        
        for image, mask in zip(batch_images, batch_masks):
            noise = torch.randn_like(image) * NOISE_MULTIPLIER*0
            image = image + noise
            image = image.squeeze()
            print(f'image.shape: {image.shape}')
            image = (image*255).numpy().astype('uint8')
            mask = (mask*255).numpy().astype('uint8')            
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.tight_layout()
            plt.show()
        break

