# """
# From https://www.kaggle.com/code/limitz/pytorch-dataset-with-volumetric-augmentations/notebook
# """
# # import inline # inline is a python file in the same directory but its saying no module named inline
# import os
# import math
# import glob
# import tqdm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as T
# import torchvision.transforms.functional as TF
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def load_volume(dataset, labeled=True, slice_range=None):
#     ''' Load slices into a volume. Keeps the memory requirement
#         as low as possible by using uint8 and uint16 in CPU memory.
#     '''
#     if labeled:
#         path = os.path.join(dataset, "labels", "*.tif")
#     else:
#         path = os.path.join(dataset, "images", "*.tif")
    
#     dataset = sorted(glob.glob(path))
#     volume = None
#     target = None
#     keys = []
#     offset = 0 if slice_range is None else slice_range[0]
#     depth = len(dataset) if slice_range is None else slice_range[1]-slice_range[0]
    
#     for z, path in enumerate(tqdm.tqdm(dataset)):
#         if slice_range is not None:
#             if z < slice_range[0]: continue
#             if z >= slice_range[1]: continue
        
#         parts = path.split(os.path.sep)
#         key = parts[-3] + "_" + parts[-1].split(".")[0]
#         keys.append(key)
                
#         if labeled:
#             label = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
#             label = np.array(label,dtype=np.uint8)
#             if target is None:
#                 target = np.zeros((1,depth, *label.shape[-2:]), dtype=np.uint8)
#             target[:,z-offset] = label
        
#         path = path.replace("/labels/","/images/")
#         path = path.replace("/kidney_3_dense/","/kidney_3_sparse/")
#         image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
#         image = np.array(image,dtype=np.uint16)
        
#         if volume is None:
#             volume = np.zeros((1,depth, *image.shape[-2:]), dtype=np.uint16)
#         volume[:,z-offset] = image
    
#     return volume, target, keys

# class RandomVolumetricDataset(torch.utils.data.Dataset):
#     ''' Dataset for segmentation of a sparse class. Keeps
#         track of positive samples and favors samples that
#         contain a positive sample.
#         WARNING: do not use in a distributed setting.
#     '''
#     def __init__(self, datasets, shape=(256,256,256), length=1000, transform=None):
#         self.volumes = []
#         self.targets = []
#         self.length = length
#         self.shape = shape
#         self.transform = transform
#         self.nonzero = []
        
#         for dataset in datasets:
#             print("loading volume", dataset)
#             volume, target, _ = load_volume(dataset)
#             self.volumes.append(volume)
#             self.targets.append(target)
#             self.nonzero.append(np.argwhere(target > 0))
        
#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         vidx = torch.randint(len(self.volumes), (1,)).item()
#         volume = self.volumes[vidx]
#         target = self.targets[vidx]
#         nonzero = self.nonzero[vidx]
#         random = torch.rand(1)
        
#         if random > 0.9:
#             # Load a random subvolume
#             z,y,x = torch.randint(volume.shape[-3]-self.shape[-3], (1,)).item(), \
#                     torch.randint(volume.shape[-2]-self.shape[-2], (1,)).item(), \
#                     torch.randint(volume.shape[-1]-self.shape[-1], (1,)).item()
#         else:
#             # Load a subvolume containing a random sample
#             idx = torch.randint(len(nonzero), (1,)).item()
#             c,z,y,x = nonzero[idx]
            
#             z += torch.randint(self.shape[-3],(1,)).sub(self.shape[-3]//2).item()
#             y += torch.randint(self.shape[-2],(1,)).sub(self.shape[-2]//2).item()
#             x += torch.randint(self.shape[-1],(1,)).sub(self.shape[-1]//2).item()
            
#             z = min(max(0,z+self.shape[-3]//2), volume.shape[-3]-self.shape[-3])
#             y = min(max(0,y+self.shape[-2]//2), volume.shape[-2]-self.shape[-2])
#             x = min(max(0,x+self.shape[-3]//2), volume.shape[-1]-self.shape[-1])
            
#         volume = volume[:,z:z+self.shape[-3], y:y+self.shape[-2], x:x+self.shape[-1]]
#         target = target[:,z:z+self.shape[-3], y:y+self.shape[-2], x:x+self.shape[-1]]

#         volume = torch.from_numpy((volume/65536).astype(np.float32))
#         target = torch.from_numpy(target > 0).float()
#         if self.transform is not None:
#             rng = torch.get_rng_state()
#             volume = self.transform(volume)
#             torch.set_rng_state(rng)
#             target = self.transform(target)
        
#         return volume, target

# # The augmentations

# class RandomRotationNd(nn.Module):
#     ''' This augmentation first permutes the dimensions as an initial rotation
#         to select the rotation axis, then rotates around the (fixed) z axis. 
#         The result is zoomed in to remove empty space and finally permuted 
#         once more to move randomize the rotation axis.
#     '''
#     def __init__(self, dims):
#         super().__init__()
#         self.dims = dims

#     def forward(self, x):
#         angle = torch.rand(1).item() * 360
#         keep = torch.arange(x.dim() - self.dims)
#         perm = -torch.randperm(self.dims)-1
#         x = x.clone().permute(*[k.item() for k in keep], *[p.item() for p in perm])
#         rad = math.pi * angle / 180
#         scale = abs(math.sin(rad)) + abs(math.cos(rad))
#         for i in range(0, x.shape[-3],8): # presumptuous
#             v = x[...,i:i+8,:,:]
#             w = v.view(-1, *v.shape[-3:])
#             w = TF.rotate(w, angle)
#             v = w.view(*v.shape)
#             x[...,i:i+8,:,:] = v
#         s = x.shape
#         x = F.interpolate(x, scale_factor=scale, mode="bilinear")
#         x = TF.center_crop(x, s[-2:])
#         perm = -torch.randperm(self.dims)-1
#         x = x.permute(*[k.item() for k in keep], *[p.item() for p in perm])
#         return x

# class RandomRot90Nd(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.dims = dims

#     def forward(self, x):
#         dims = -torch.randperm(self.dims)[:2]-1
#         dims = [d.item() for d in dims]
#         rot = torch.randint(4, (1,)).item()
#         return x.rot90(rot, dims)

# class RandomPermuteNd(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.dims = dims

#     def forward(self, x):
#         perm = -torch.randperm(self.dims)-1
#         keep = torch.arange(x.dim() - self.dims)
#         return x.permute(*[k.item() for k in keep], *[p.item() for p in perm])

# class RandomFlipNd(nn.Module):
#     def  __init__(self, dims, p=0.5):
#         super().__init__()
#         self.dims = dims
#         self.p = p
        
#     def forward(self, x):
#         for i in range(self.dims):
#             if torch.rand(1) < self.p:
#                 x = x.flip(-i-1)
#         return x

# class ToDevice(nn.Module):
#     ''' Sometimes it helps to move the tensor to the gpu before augmentations like
#         rotation. Note however that you need to set num_workers to 0 in the dataloader
#     '''
#     def __init__(self, device=None):
#         super().__init__()
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

#     def forward(self, x):
#         return x.to(self.device)

# transform = T.Compose((ToDevice(), RandomRotationNd(3), RandomFlipNd(3)))
# ds = RandomVolumetricDataset([
#     "data/train/kidney_1_dense",
#     "data/train/kidney_3_dense"
#     ], length=1000, shape=(512,512,512),
#     transform=transform)

# print("Each sample returned from the dataset is random and augmented")
# torch.manual_seed(125)
# volume, target = ds[0] # ^ irregardless of idx, which is why it 
#                        # doesn't work in distributed settings 

# volume = volume.sub(volume.mean()).div(volume.std().add(1e-5)) # normalize
# # inline.plot(torch.stack((inline.disp(volume[0]), inline.disp(target[0]))), width=10)

# # plot slices from each axis of the volume
# slices = torch.stack([
#     volume[:,volume.shape[-3]//2,:,:].cpu(),
#     volume[:,:,volume.shape[-2]//2,:].cpu(),
#     volume[:,:,:,volume.shape[-1]//2].cpu(),
# ])

# plt.figure(figsize=(10,10))
# plt.imshow(torchvision.utils.make_grid(slices, nrow=3).permute(1,2,0))
# plt.show()

# # print("For show: more augmentation of the same subvolume")
# # # Showing the same subvolume, with random rotations
# # rot = RandomRotationNd(3)
# # rng = torch.get_rng_state()
# # volumes = torch.stack([rot(volume)[0] for _ in range(8)])
# # torch.set_rng_state(rng)
# # targets = torch.stack([rot(target)[0] for _ in range(8)])
# # # inline.plot(volumes.mul(0.288).add(0.5)[:,[100]])
# # # inline.plot(volumes)
# # # inline.plot(targets)



#############################################################################################################


import torch
import numpy as np
import random
import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from global_params import *

def load_volume(path, slice_shape=(512, 512), depth=512):
    # Assuming the path contains a series of image files
    # Load the images into a 3D volume
    files = sorted(glob.glob(os.path.join(path, "*"+IMG_FILE_EXT)))
    volume = np.zeros((depth, *slice_shape), dtype=np.float32)
    
    for i, file in enumerate(files[:depth]):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, slice_shape)
        volume[i] = img
    
    return volume

class RandomFlip3D(transforms.RandomHorizontalFlip):
    """Randomly flip a 3D volume."""
    def forward(self, img):
        if torch.rand(1) < self.p:
            return img.flip(-1)  # Assuming the last dimension is the one to be flipped
        return img

class RandomRotation3D(transforms.RandomRotation):
    """Randomly rotate a 3D volume."""
    def forward(self, img):
        angle = self.get_params(self.degrees)
        return img.rotate(angle)  # Implement a rotate function for 3D volume

class KidneyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # Assuming each sub-directory in root_dir is a separate scan
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        scan_path = self.root_dir
        volume = load_volume(scan_path)

        if self.transform:
            volume = self.transform(volume)

        # Extract slices from the volume
        slices = []
        for i in range(volume.shape[0]):  # Iterate through the depth
            slice_2d = volume[i]
            slices.append(slice_2d)

        return torch.tensor(slices, dtype=torch.float32)

# Usage
transform = transforms.Compose([
    RandomFlip3D(p=0.5),
    RandomRotation3D(degrees=(0, 360))
])

import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

# Assuming the KidneyDataset class and necessary imports are already defined above

def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib."""
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

# Set up the dataset and data loader
transform = Compose([ToTensor()])  # Add any required transforms here
dataset = KidneyDataset(root_dir=VAL_IMG_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Fetch and display a few random images
num_images_to_show = 4
images = []
for i, data in enumerate(dataloader):
    volume, _ = data  # Assuming each item in the dataset is a (volume, target) tuple
    # Select a random slice from the volume
    random_slice_idx = torch.randint(0, volume.shape[1], (1,)).item()
    slice_2d = volume[0, random_slice_idx, :, :].numpy()
    images.append(slice_2d)
    if i >= num_images_to_show - 1:
        break

# Plot the images
show_images(images, cols=2)