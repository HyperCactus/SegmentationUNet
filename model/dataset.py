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


class SegmenTrainDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, transform=None, 
                 img_file_ext='.tif', mask_file_ext='.tif'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.tensorize = ToTensorV2()
        # self.images = [img for img in os.listdir(self.image_dir) if img.endswith(file_ext)]

        self.image_files = glob(os.path.join(self.image_dir, f'*{img_file_ext}'))
        self.mask_files = glob(os.path.join(self.mask_dir, f'*{mask_file_ext}'))
        
        assert len(self.image_files) == len(self.mask_files), \
            f'Number of images and masks do not match. Found {len(self.image_files)} images and {len(self.mask_files)} masks.'
        # self.image_list = os.listdir(self.image_dir)
        # self.image_list = [os.path.join(self.image_dir, i) for i in self.image_list if i.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        # images and masks are 1 channel greyscale
        image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)
        image = image / 255.0 # normalize images to be between 0 and 1
        mask = cv2.imread(self.mask_files[index], cv2.IMREAD_GRAYSCALE)
        mask[mask == 255.0] = 1.0 # convert all 255 values to 1.0 to make it a binary mask
        
        if self.transform is not None:    # IMPLEMENT TRANSFORMS HERE
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        if not isinstance(image, torch.Tensor):
            image = self.tensorize(image=image)['image']
        if not isinstance(mask, torch.Tensor):
            mask = self.tensorize(image=mask)['image']
        
        # the datatype of the images and masks are float32
        image = image.type(torch.float32)
        mask = mask.type(torch.float32)
        
        return image, mask

def create_loader(image_dir, mask_dir, batch_size, 
                  transform=None, shuffle=False):
    dataset = SegmenTrainDataset(image_dir, mask_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

VAL_TRANSFORMS = A.Compose([
    A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=True),  # High pass filter
    ToTensorV2()
])

TRAIN_TRANSFORMS = A.Compose([
    A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, always_apply=True),
    A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit=(-0.1, 0.4), rotate_limit=15, shift_limit=0.1, p=0.8, border_mode=0),
    A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
    A.OneOf(
        [
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.7,
    ),
    A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=True),  # High pass filter
])

VAL_LOADER = create_loader(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, transform=VAL_TRANSFORMS)

# # testing the dataset:
# for batch_idx, (batch_images, batch_masks) in enumerate(VAL_LOADER):
#     # print(f'BATCH {batch_idx+1}')
#     if batch_idx < 200/BATCH_SIZE:
#         continue
#     if batch_idx > (200+BATCH_SIZE)/BATCH_SIZE:
#         break
#     print("Batch", batch_idx + 1)
#     print("Image batch shape:", batch_images.shape)
#     print("Mask batch shape:", batch_masks.shape)
    
#     for image, mask in zip(batch_images, batch_masks):
       
#         # image = image.permute((1, 2, 0)).numpy()*255.0;
#         print(f'image.shape presqueese: {image.shape}')
#         image = image.squeeze().numpy()*255.0
#         # image = image.numpy()*255.0
#         print(f'image.shape postsqueese: {image.shape}')
#         image = image.astype('uint8')
#         mask = (mask*255).numpy().astype('uint8')
        
#         # image_filename = os.path.basename(image_path)
#         # mask_filename = os.path.basename(mask_path)
        
#         plt.figure(figsize=(10, 5))
        
#         plt.subplot(1, 2, 1)
#         plt.imshow(image, cmap='gray')
#         # plt.title(f"Original Image - {image_filename}")
        
#         plt.subplot(1, 2, 2)
#         plt.imshow(mask, cmap='gray')
#         # plt.title(f"Mask Image - {mask_filename}")
        
#         plt.tight_layout()
#         plt.show()
#     break


# # ========================================================================================================
# """
# Code copied from https://www.kaggle.com/code/aniketkolte04/sennet-hoa-seg-pytorch-attention-gated-unet
# pytorch dataset for the challenge.
# """
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# import tifffile as tiff
# import cv2
# import torch.nn as nn
# import albumentations as A
# import numpy as np
# import os
# import time
# import torch.nn.functional as F
# from PIL import Image
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from global_params import *
# from sklearn.model_selection import train_test_split

# # project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # print(f'project_dir: {project_dir}')
# # os.chdir(project_dir) # change to project directory

# # images_path = os.path.join(base_path, dataset, 'images')
# # labels_path = os.path.join(base_path, dataset, 'labels')

# # image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.tif')])
# # label_files = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.tif')])

# # def show_images(images,titles= None, cmap='gray'):
# #     n = len(images)
# #     fig, axes = plt.subplots(1, n, figsize=(20, 10))
# #     if not isinstance(axes, np.ndarray):
# #         axes = [axes]
# #     for idx, ax in enumerate(axes):
# #         ax.imshow(images[idx], cmap=cmap)
# #         if titles:
# #             ax.set_title(titles[idx])
# #         ax.axis('off')
# #     plt.tight_layout()
# #     plt.show()

# # first_image = tiff.imread(image_files[981])
# # first_label = tiff.imread(label_files[981])

# # show_images([first_image, first_label], titles=['First Image', 'First Label'])

# class CustomDataset(Dataset):
#     def __init__(self, image_files, mask_files, 
#                  input_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
#                  augmentation_transforms=None):
#         self.image_files = image_files
#         self.mask_files = mask_files
#         self.input_size = input_size
#         self.augmentation_transforms = augmentation_transforms

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
       
#         image_path = self.image_files[idx]
#         mask_path = self.mask_files[idx]

#         image = preprocess_image(image_path)
#         mask = preprocess_mask(mask_path)

#         if self.augmentation_transforms:
#             image, mask = self.augmentation_transforms(image, mask)

#         return image, mask

# class UsageDataset(Dataset):
#     def __init__(self, image_files, 
#                  input_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
#                  augmentation_transforms=None):
#         self.image_files = image_files
#         self.input_size = input_size
#         self.augmentation_transforms = augmentation_transforms

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
       
#         image_path = self.image_files[idx]

#         image, orig_size = preprocess_image(image_path, return_size=True)
#         # orig_size = image.shape

#         if self.augmentation_transforms:
#             image = self.augmentation_transforms(image)

#         return image, torch.tensor(np.array([orig_size[0], orig_size[1]]))


# def preprocess_image(path, return_size=False):
    
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     # print(f'fresh process image img.shape: {img.shape}')
#     img = np.tile(img[...,None],[1, 1, 3]) 
#     img = img.astype('float32') 

#     # scaling to 0-1
#     mx = np.max(img)
#     if mx:
#         img/=mx
    
#     orig_size = img.shape
    
#     # print(f'process image img.shape: {img.shape}')
#     img = np.transpose(img, (2, 0, 1))
#     img_ten = torch.tensor(img)
#     if return_size:
#         return img_ten, orig_size
#     else:
#         return img_ten

# def preprocess_mask(path):
    
#     msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     msk = msk.astype('float32')
#     msk/=255.0
#     msk_ten = torch.tensor(msk)
    
#     return msk_ten

# def augment_image(image, mask):
    
#     image_np = image.permute(1, 2, 0).numpy()
#     mask_np = mask.numpy()

#     transform = A.Compose([
#         A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, always_apply=True),
#         A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.ShiftScaleRotate(scale_limit=(-0.1, 0.4), rotate_limit=15, shift_limit=0.1, p=0.8, border_mode=0),
#         A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
#         A.OneOf(
#             [
#                 A.Blur(blur_limit=3, p=1),
#                 A.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.7,
#         ),
#         A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=True),  # High pass filter
#     ])

#     augmented = transform(image=image_np, mask=mask_np)
#     augmented_image, augmented_mask = augmented['image'], augmented['mask']

#     augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)
#     augmented_mask = torch.tensor(augmented_mask, dtype=torch.float32)

#     return augmented_image, augmented_mask

# def val_transform(image, mask):
    
#     image_np = image.permute(1, 2, 0).numpy()
#     mask_np = mask.numpy()

#     transform = A.Compose([
#         A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
#         A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=True),  # High pass filter
#     ])

#     augmented = transform(image=image_np, mask=mask_np)
#     augmented_image, augmented_mask = augmented['image'], augmented['mask']

#     augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)
#     augmented_mask = torch.tensor(augmented_mask, dtype=torch.float32)

#     return augmented_image, augmented_mask

# def create_loader(image_files, mask_files, batch_size, 
#                   augmentations=None, shuffle=False):
    
#     dataset = CustomDataset(image_files, mask_files, augmentation_transforms=augmentations)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# def create_test_loader(image_files, batch_size, 
#                   augmentations=None, shuffle=False):
    
#     dataset = UsageDataset(image_files, augmentation_transforms=augmentations)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    

# #-------------------------- Test the dataset --------------------------#
# image_files = []
# label_files = []
# for dataset in datasets:
#     images_path = os.path.join(BASE_PATH, dataset, 'images')
#     labels_path = os.path.join(BASE_PATH, dataset, 'labels')

#     image_files.extend(sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.tif')]))
#     label_files.extend(sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.tif')]))

# # images_path = os.path.join(base_path, dataset, 'images')
# # labels_path = os.path.join(base_path, dataset, 'labels')

# # image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.tif')])
# # label_files = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.tif')])

# train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
#     image_files, label_files, test_size=0.1, random_state=42)

# # testing_path = 'data_downsampled512/train/test_output'
# # testing_img_files = sorted([os.path.join(testing_path, 'images', f) for f in os.listdir(testing_path+'/images') if f.endswith('.tif')])
# # testing_mask_files = sorted([os.path.join(testing_path, 'labels', f) for f in os.listdir(testing_path+'/labels') if f.endswith('.tif')])
# # testing_mask_files = train_mask_files[:len(testing_img_files)]

# train_dataset = CustomDataset(train_image_files, train_mask_files, augmentation_transforms=augment_image)
# val_dataset = CustomDataset(val_image_files, val_mask_files, augmentation_transforms=val_transform)
# # test_of_dataset = CustomDataset(testing_img_files, testing_mask_files, augmentation_transforms=val_transform)

# TRAIN_LOADER = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# VAL_LOADER = DataLoader(val_dataset, batch_size=1, shuffle=False)
# # testing_loader = DataLoader(test_of_dataset, batch_size=1, shuffle=True)

# # for batch_idx, (batch_images, batch_masks) in enumerate(VAL_LOADER):
# #     print("Batch", batch_idx + 1)
# #     print("Image batch shape:", batch_images.shape)
# #     print("Mask batch shape:", batch_masks.shape)


# # for batch_idx, (batch_images, batch_masks) in enumerate(VAL_LOADER):
# #     print("Batch", batch_idx + 1)
# #     print("Image batch shape:", batch_images.shape)
# #     print("Mask batch shape:", batch_masks.shape)
    
# #     for image, mask, image_path, mask_path in zip(batch_images, batch_masks, train_image_files, train_mask_files):
       
# #         image = image.permute((1, 2, 0)).numpy()*255.0
# #         image = image.astype('uint8')
# #         mask = (mask*255).numpy().astype('uint8')
        
# #         image_filename = os.path.basename(image_path)
# #         mask_filename = os.path.basename(mask_path)
        
# #         plt.figure(figsize=(15, 10))
        
# #         plt.subplot(2, 4, 1)
# #         plt.imshow(image, cmap='gray')
# #         plt.title(f"Original Image - {image_filename}")
        
# #         plt.subplot(2, 4, 2)
# #         plt.imshow(mask, cmap='gray')
# #         plt.title(f"Mask Image - {mask_filename}")
        
# #         plt.tight_layout()
# #         plt.show()
# #     break

# # for batch_idx, (batch_images, batch_masks) in enumerate(VAL_LOADER):
# #     print("Batch", batch_idx + 1)
# #     print("Image batch shape:", batch_images.shape)
# #     print("Mask batch shape:", batch_masks.shape)
    
# #     for image, mask, image_path, mask_path in zip(batch_images, batch_masks, train_image_files, train_mask_files):
       
# #         image = image.permute((1, 2, 0)).numpy()*255.0
# #         image = image.astype('uint8')
# #         mask = (mask*255).numpy().astype('uint8')
        
# #         image_filename = os.path.basename(image_path)
# #         mask_filename = os.path.basename(mask_path)
        
# #         plt.figure(figsize=(15, 10))
        
# #         plt.subplot(2, 4, 1)
# #         plt.imshow(image, cmap='gray')
# #         plt.title(f"Original Image - {image_filename}")
        
# #         plt.subplot(2, 4, 2)
# #         plt.imshow(mask, cmap='gray')
# #         plt.title(f"Mask Image - {mask_filename}")
        
# #         plt.tight_layout()
# #         plt.show()
# #     break
