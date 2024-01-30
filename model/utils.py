"""
Some helper functions for visualization, saving/loading checkpoints and calculating the dice score.
"""
# Copied from COMP3710 report

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from torchvision.transforms import PILToTensor
import random
import os
import time
import math
from tqdm import tqdm
import torch
import torch.nn as nn
# from costom_loss import IoULoss
from global_params import *
from dataset import create_loader, preprocess_image, preprocess_mask
from helper import fast_compute_surface_dice_score_from_tensor
from glob import glob
from torch import Tensor, einsum
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
from scipy.ndimage import distance_transform_edt as eucl_distance

val_transform = None

# The below class is from https://github.com/hubutui/DiceLoss-PyTorch and was modified
# It is a loss function based on the Dice score.
class BinaryDiceScore(nn.Module):
    """Dice score of binary class
    Args:
        smooth: A float number to smooth score, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Dice score tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-6, p=2, reduction='sum'):
        super(BinaryDiceScore, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        dice_score = num / den

        if self.reduction == 'mean':
            return dice_score.mean()
        elif self.reduction == 'sum':
            return dice_score.sum()
        elif self.reduction == 'none':
            return dice_score
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

# def evaluate(model, loader, device='cuda', criterion=None,
#              threshold=0.5, verbose=False, leave_on_train=False,
#              score_type='dice'):
#     """
#     Calculates the average dice score over the entire dataloader
#     """
#     print('>>> Calculating Dice Score')
    
#     model.eval()
#     dice_scores = []
#     iou_scores = []
#     iou_loss = IoULoss()
#     criterion = BinaryDiceScore() if criterion is None else criterion
#     loop = tqdm(loader) if verbose else loader
    
#     with torch.no_grad():
#         for _, (x, y) in enumerate(loop):
#             x = x.to(device)
#             y = y.to(device)
#             preds = model(x)
#             preds = (preds > threshold).float() # convert to binary mask
#             dice_scores.append(criterion(preds, y).item())
#             iou_scores.append(1 - iou_loss(preds, y).item())
#     if leave_on_train:
#         model.train()
    
#     mean_dice_score = np.mean(dice_scores)
#     mean_iou_score = np.mean(iou_scores)
#     if score_type == 'dice':
#         return mean_dice_score
#     elif score_type == 'iou':
#         return mean_iou_score
#     elif score_type == 'both':
#         return mean_dice_score, mean_iou_score

def save_checkpoint(state, filename='checkpoints/checkpoint.pth.tar'):
    """
    Saves the model and optimizer state dicts to a checkpoint file
    """
    print('>>> Saving checkpoint')
    # os.makedirs('checkpoints', exist_ok=True)
    torch.save(state, filename)
    print('>>> Checkpoint saved')

def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads the model and optimizer state dicts from a checkpoint file
    """
    print('>>> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print('>>> Checkpoint loaded')

def calc_dice_score(model, dataloader, device='cuda', verbose=False):
    """
    Calculates the average dice score over the entire dataloader
    """
    model.eval()
    print('>>> Calculating Dice Score')
    with torch.no_grad():
        dice_score = 0
        for _, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = (preds > 0.5).float() # convert to binary mask
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8) # add 1e-8 to avoid division by 0
    model.train()
    dice_score = dice_score / len(dataloader)
    dice_score = dice_score.item()
    dice_score = np.round(dice_score, 4)
    return dice_score

def save_predictions_as_imgs(loader, model, num, folder='saved_images/', device='cuda', verbose=True):
    """
    Saves the predictions from the model as images in the folder
    """
    preds_path = f'{folder}preds/'
    masks_path = f'{folder}masks/'
    orig_path = f'{folder}orig/'
    os.makedirs(preds_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    os.makedirs(orig_path, exist_ok=True)
    
    model.eval()
    print('>>> Generating and saving predictions') if verbose else None
    loop = tqdm(enumerate(loader), total=num, leave=False) if verbose else enumerate(loader)
    with torch.no_grad():
        # for idx, (x, y) in enumerate(loader):
        for idx, (x, y) in loop:
            x = x.to(device)
            y = y.to(device) # add 1 channel to mask
            preds = torch.sigmoid(model(x))
            preds = (preds > PREDICTION_THRESHOLD).float()
            torchvision.utils.save_image(preds, f'{preds_path}pred_{idx+1}.png')
            torchvision.utils.save_image(y.unsqueeze(1), f'{masks_path}mask_{idx+1}.png')
            torchvision.utils.save_image(x, f'{orig_path}orig_{idx+1}.png')
            if idx == num-1:
                break
    model.train()

def plot_prediction(ind=0, folder='saved_images'):
    """
    Assumes the folder contains the following subfolders:
    preds, masks, orig.
    Plots the original image at ind, the mask, and the prediction side by side with labels.
    """
    preds_path = f'{folder}/preds/'
    masks_path = f'{folder}/masks/'
    orig_path = f'{folder}/orig/'
    
    preds = os.listdir(preds_path)
    masks = os.listdir(masks_path)
    origs = os.listdir(orig_path)
    
    if ind >= len(preds):
        raise IndexError(f'Index {ind} out of range. There are {len(preds)} predictions.')
    
    # plot only one image from each folder
    pred = Image.open(preds_path + preds[ind])
    mask = Image.open(masks_path + masks[ind])
    orig = Image.open(orig_path + origs[ind])
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    axs[0].imshow(orig)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(mask)
    axs[1].set_title('Mask')
    axs[1].axis('off')
    axs[2].imshow(pred)
    axs[2].set_title('Prediction')
    axs[2].axis('off')
    plt.show()

def eval_and_plot(model, loader, device='cuda', 
                         save_location='examples/', 
                         shuffle=True, n_images=6, 
                         threshold=0.5, verbose=True):
    """
    Assumes the loader has batch size 1.
    """
    model.eval()
    dice_scores = []
    criterion = BinaryDiceScore()# if criterion is None else criterion
    plot_images = []
    loop = tqdm(loader) if verbose else loader
    
    # choose n_images random indices from the loader
    if shuffle:
        indices = random.sample(range(len(loader)), n_images)
    else:
        indices = range(n_images)
    
    with torch.no_grad():
        for ind, (x, y) in enumerate(loop):
            img = x.to(device)
            mask = y.to(device)
            pred = model(img)
            pred = (pred > threshold).float() # convert to binary mask
            dice_scores.append(criterion(pred, mask).item())
            
            if ind in indices:
                # need to reshape the image and mask to not have batch dimension
                img = img.squeeze(0)
                mask = mask.squeeze(1)
                pred = pred.squeeze(1)
                # print(f'IMG SHAPE: {img.shape}, MASK SHAPE: {mask.shape}')
                
                
                # plot_images.append(img)
                plot_images.append(mask)
                plot_images.append(pred)

    # if leave_on_train:
    # model.train()
    
    examples = torchvision.utils.make_grid(plot_images, nrow=2)
    examples = examples.cpu()
    # plot the examples
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(examples.permute(1, 2, 0))
    ax.axis('off')
    ax.set_title('Examples, (mask, prediction)')
    plt.show()
    # save the examples to the save_location
    file = f'{save_location}examples.png'
    torchvision.utils.save_image(examples, file)
    
    mean_dice_score = np.mean(dice_scores)
    return mean_dice_score

def plot_samples(num, folder='saved_images', 
                 include_image=True, shuffle=True, 
                 title='Samples'):
    """
    Assumes the folder contains the following subfolders:
    preds, masks, orig.
    Plots num predictions side by side with the masks and original images if include_image=True.
    Similar to plot_prediction but for num images.
    """
    preds_path = f'{folder}/preds/'
    masks_path = f'{folder}/masks/'
    orig_path = f'{folder}/orig/'
    
    preds = os.listdir(preds_path)
    masks = os.listdir(masks_path)
    origs = os.listdir(orig_path)
    
    if num > len(preds):
        raise ValueError(f'num = {num} out of range. There are {len(preds)} predictions.')
    
    if shuffle:
        indices = random.sample(range(len(preds)), num)
    else:
        indices = range(num)
    
    # use the torchvision function plot grid to plot the images in 3 columns: original, mask, prediction
    if include_image:
        fig, axs = plt.subplots(num, 3, figsize=(20, 10*num))
        image_list = []
        for i, ind in enumerate(indices):
            # image_list.append(Image.open(preds_path + preds[ind]))
            # image_list.append(Image.open(masks_path + masks[ind]))
            # image_list.append(Image.open(orig_path + origs[ind]))
            
            pred = Image.open(preds_path + preds[ind])
            mask = Image.open(masks_path + masks[ind])
            orig = Image.open(orig_path + origs[ind])
            axs[i, 0].imshow(orig)
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(mask)
            axs[i, 1].set_title('Mask')
            axs[i, 1].axis('off')
            axs[i, 2].imshow(pred)
            axs[i, 2].set_title('Prediction')
            axs[i, 2].axis('off')
        fig.suptitle(title)
        plt.show()
        # torchvision.utils.make_grid(image_list, nrow=3).show()
    else:
        fig, axs = plt.subplots(num, 2, figsize=(20, 10*num))
        for i, ind in enumerate(indices):
            # image_list.append(Image.open(preds_path + preds[ind]))
            # image_list.append(Image.open(masks_path + masks[ind]))
            
            pred = Image.open(preds_path + preds[ind])
            mask = Image.open(masks_path + masks[ind])
            axs[i, 0].imshow(mask)
            axs[i, 0].set_title('Mask')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(pred)
            axs[i, 1].set_title('Prediction')
            axs[i, 1].axis('off')
        fig.suptitle(title)
        plt.show()
        
        # torchvision.utils.make_grid(image_list, nrow=2).show()

def plot_samples_mask_overlay(dataset, n=12):
    """
    Plots n samples from the dataset
    """
    fig, axs = plt.subplots(2, n//2, figsize=(20, 10))
    for i in range(n):
        img, mask = dataset[i]
        axs[i//6, i%6].imshow(img)
        axs[i//6, i%6].imshow(mask, alpha=0.3) # overlay mask
        axs[i//6, i%6].axis('off')
        axs[i//6, i%6].set_title('Sample #{}'.format(i))
    plt.show()

def print_progress(start_time, epoch, num_epochs):
    """
    Estimates the time remaining in the training loop and prints the progress
    """
    elapsed_time = time.time() - start_time
    average_time_per_epoch = elapsed_time / (epoch + 1)
    remaining_time = average_time_per_epoch * (num_epochs - epoch - 1)
    # convert elapsed time to days, hours and minutes
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    # convert to days, hours, minutes, seconds
    days = remaining_time // (24 * 3600)
    remaining_time = remaining_time % (24 * 3600)
    hours = remaining_time // 3600
    remaining_time %= 3600
    minutes = remaining_time // 60
    remaining_time %= 60
    print(f"""Epoch [{epoch+1}/{num_epochs}] completed. Time elapsed: {elapsed_time}. \
seconds. Time remaining: {days:.0f} days, {hours:.0f} hours, \
{minutes:.0f} minutes.""", end='\r')



def dice_coeff(prediction, target, threshold=0.5):

    mask = np.zeros_like(prediction)
    mask[prediction >= threshold] = 1

    inter = np.sum(mask * target)
    union = np.sum(mask) + np.sum(target)
    epsilon = 1e-6
    result = np.mean(2 * inter / (union + epsilon))
    return result


# The below functions are from https://www.kaggle.com/code/jirkaborovec/sennet-hoa-rle-decode-encode-demo-submission

def rle_decode(mask_rle: str, img_shape: tuple = None) -> np.ndarray:
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    assert len(starts) == len(lengths)
    ends = starts + lengths
    img = np.zeros((np.product(img_shape),), dtype=np.uint8)
    for begin, end in zip(starts, ends):
        img[begin:end] = 1
    # https://stackoverflow.com/a/46574906/4521646
    img.shape = img_shape
    return img


def rle_encode(mask, bg = 0) -> dict:
    vec = mask.flatten()
    nb = len(vec)
    where = np.flatnonzero
    starts = np.r_[0, where(~np.isclose(vec[1:], vec[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, nb])
    values = vec[starts]
    assert len(starts) == len(lengths) == len(values)
    rle = []
    for start, length, val in zip(starts, lengths, values):
        if val == bg:
            continue
        rle += [str(start), length]
    # post-processing
    return " ".join(map(str, rle))

def create_rle_df(kidney_dirs: [str], 
                  subdir_name: str = 'preds',
                  img_size=(512, 512), 
                  resize=True,
                  orig_path='/kaggle/input/blood-vessel-segmentation/test/'):
    """
    Creates a dataframe with the image ids and the predicted masks
    """
    if type(kidney_dirs) == str:
        kidney_dirs = [kidney_dirs]
    
    df = pd.DataFrame(columns=['id', 'rle'])
    # df.set_index('id', inplace=True)
        
    for kidney_dir in kidney_dirs:
        assert os.path.exists(kidney_dir), f'{kidney_dir} does not exist'

        if resize:
            example_path = os.path.join(orig_path, kidney_name, 'images', '0000.tif')
            assert os.path.exists(example_path), f'{example_path} does not exist'
            orig_size = cv2.imread(os.path.join(orig_path, os.path.basename(kidney_dir), 
                                                'images', '0001.tif')).shape[:2]
            img_size = (orig_size[1], orig_size[0])

        kidney_name = os.path.basename(kidney_dir)
        masks = sorted([os.path.join(kidney_dir, subdir_name, f) for f in 
                        os.listdir(os.path.join(kidney_dir, subdir_name)) if f.endswith('.tif')])
        
        for mask_file in masks:
            mask_name = os.path.basename(mask_file)
            # mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask = Image.open(mask_file).convert('L').resize(img_size, resample=Image.BOX)

            # display a mask to check if it is correct
            plt.imshow(mask)

            mask = np.array(mask, dtype=np.float32)

            print(f'MASK SHAPE: {mask.shape}, MASK MAX: {mask.max()}')

            

            # # ToDo: rezise the mask in a smart way
            # # the mask may need to be upscaled or downscaled
            # mask = mask.resize(img_size, )
            
            id = f'{kidney_name}_{mask_name[:-4]}'
            rle = rle_encode(mask)
            df.loc[len(df)] = [id, rle]
        
        df['width'] = [img_size[0]] * len(df)
        df['height'] = [img_size[1]] * len(df)
    
    return df

def _predict_transform(image, mask):
    
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    transform = A.Compose([
        A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=True),  # High pass filter
    ])

    augmented = transform(image=image_np, mask=mask_np)
    augmented_image, augmented_mask = augmented['image'], augmented['mask']

    augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)
    augmented_mask = torch.tensor(augmented_mask, dtype=torch.float32)

    return augmented_image, augmented_mask

def save_predictions(kidney_dirs, model, num='all', min_idx=0,
                     folder='saved_images/', device='cuda', 
                     verbose=True, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                     keep_size=False):
    """
    Saves the predictions from the model as images in the folder
    """
    if type(kidney_dirs) == str:
        kidney_dirs = [kidney_dirs]

    for kidney_dir in kidney_dirs:
        assert os.path.exists(kidney_dir), f'{kidney_dir} does not exist'
        
        kidney_name = os.path.basename(kidney_dir)
        
        preds_path = os.path.join(folder, kidney_name, 'preds')
        masks_path = os.path.join(folder, kidney_name, 'labels')
        images_path = os.path.join(folder, kidney_name, 'images')
        os.makedirs(preds_path, exist_ok=True)
        os.makedirs(masks_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        
        masks = sorted([os.path.join(kidney_dir, 'labels', f) for f in 
                        os.listdir(os.path.join(kidney_dir, 'labels')) if f.endswith('.tif')])
        images = sorted([os.path.join(kidney_dir, 'images', f) for f in 
                        os.listdir(os.path.join(kidney_dir, 'images')) if f.endswith('.tif')])
                
        if num == 'all':
            num = len(masks)
        else:
            num = min(num, len(masks))
        
        assert min_idx + num <= len(masks), f'Index out of range. There are {len(masks)} masks.'
        
        post_transforms = A.Compose([
            A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_CUBIC),
        ])
        
        if keep_size:
            loader = create_loader(images, masks, batch_size=1, augmentations=_predict_transform)
        else:
            loader = create_loader(images, masks, batch_size=1, augmentations=val_transform)
        
        model.eval()
        print('>>> Generating and saving predictions') if verbose else None
        loop = tqdm(enumerate(loader), total=num, leave=False) if verbose else enumerate(loader)
        with torch.no_grad():
            for idx, (x, y) in loop:
                if idx < min_idx:
                    continue
                
                filename = os.path.basename(masks[idx])
                x = x.to(device)
                y = y.to(device) # add 1 channel to mask
                preds = torch.sigmoid(model(x))
                preds = (preds > PREDICTION_THRESHOLD).float()

                # resize the predictions
                # preds = preds.to('cpu').numpy()
                # preds = post_transforms(image=preds)['image']
                # preds = torch.tensor(preds, dtype=torch.float32).unsqueeze(0)

                torchvision.utils.save_image(preds, os.path.join(preds_path, filename))
                torchvision.utils.save_image(y.unsqueeze(1), os.path.join(masks_path, filename))
                torchvision.utils.save_image(x, os.path.join(images_path, filename))
                if idx > min_idx + num:
                    break
        model.train()
        
def view_examples(kidney_dirs, pred_folder='preds', num=4, 
                  save_location='save_data/examples', 
                  save=False, shuffle=True):
    """
    kidney_dirs is either a string or list of strings that are the paths to the kidney directories
    each kidney directory should contain the folders images, labels and preds. 
    Creates a grid of images with the original image, the mask and the prediction.
    """
    if type(kidney_dirs) == str:
        kidney_dirs = [kidney_dirs]

    for kidney_dir in kidney_dirs:
        assert os.path.exists(kidney_dir), f'{kidney_dir} does not exist'
        
        kidney_name = os.path.basename(kidney_dir)
        
        preds_path = os.path.join(kidney_dir, pred_folder)
        masks_path = os.path.join(kidney_dir, 'labels')
        images_path = os.path.join(kidney_dir, 'images')

        mask_files = sorted([os.path.join(masks_path, f) for f in 
                        os.listdir(masks_path) if f.endswith('.tif')])
        image_files = sorted([os.path.join(images_path, f) for f in 
                        os.listdir(images_path) if f.endswith('.tif')])
        pred_files = sorted([os.path.join(preds_path, f) for f in 
                        os.listdir(preds_path) if f.endswith('.tif')])
        
        if shuffle:
            indices = random.sample(range(len(mask_files)), num)
        else:
            indices = range(num)
        
        images = []
        masks = []
        preds = []

        transform = PILToTensor()
        for i in indices:
            images.append(transform(Image.open(image_files[i])))
            masks.append(transform(Image.open(mask_files[i])))
            preds.append(transform(Image.open(pred_files[i])))

        fig_width = 20
        fir, axs = plt.subplots(1, 3, figsize=(fig_width, fig_width*num/3))
        
        imgs_ax, masks_ax, preds_ax = axs[0], axs[1], axs[2]
        images_col = torchvision.utils.make_grid(images, nrow=1)
        masks_col = torchvision.utils.make_grid(masks, nrow=1)
        preds_col = torchvision.utils.make_grid(preds, nrow=1)

        imgs_ax.imshow(images_col.permute(1, 2, 0))
        imgs_ax.axis('off')
        imgs_ax.set_title('Original Image')

        masks_ax.imshow(masks_col.permute(1, 2, 0))
        masks_ax.axis('off')
        masks_ax.set_title('Mask')

        preds_ax.imshow(preds_col.permute(1, 2, 0))
        preds_ax.axis('off')
        preds_ax.set_title('Prediction')

        fir.suptitle(f'Examples from {kidney_name}')

        if save:
            plt.savefig(f'{save_location}/{kidney_name}_examples.png')
        
        plt.show()


def batch_tiling_split(image_batch, tile_size, tiles_in_x=0, tiles_in_y=0, debug=False):
    """
    Splits a batch of images into evenly spaced tiles of size tile_size x tile_size.
    Assumes batch shape is [b, c, h, w]
    Args:
        image_batch: batch of images to split
        tile_size: size of tiles to split into
        tiles_in_x: number of tiles in x direction
        tiles_in_y: number of tiles in y direction
    Returns:
        A list of tiles
    """
    b, c, h, w = image_batch.shape

    # Calculate number of tiles if not specified
    if tiles_in_x == 0:
        tiles_in_x = math.ceil(w / tile_size)
    if tiles_in_y == 0:
        tiles_in_y = math.ceil(h / tile_size)

    # Calculate stride to evenly space tiles
    stride_x = (w - tile_size) / (tiles_in_x - 1) if tiles_in_x > 1 else 0
    stride_y = (h - tile_size) / (tiles_in_y - 1) if tiles_in_y > 1 else 0

    if debug:
        print(f'Image Size: {h}x{w}, Tile Size: {tile_size}x{tile_size}')
        print(f'Tiles in x: {tiles_in_x}, Tiles in y: {tiles_in_y}')
        print(f'Stride in x: {stride_x}, Stride in y: {stride_y}')

    tiles = []
    for i in range(tiles_in_y):
        for j in range(tiles_in_x):
            # Calculate start coordinates for the tile
            start_x = int(j * stride_x)
            start_y = int(i * stride_y)
            tile = image_batch[:, :, start_y:start_y + tile_size, start_x:start_x + tile_size]
            tiles.append(tile)

    return tiles



def recombine_tiles(tiles, original_size, tile_size, tiles_in_x, tiles_in_y):
    """
    Recombines a list of tiled predictions into a single prediction batch.
    Adjusts edge tiles to fit the original image dimensions.
    Args:
        tiles: list of tiled predictions
        original_size: tuple, size of the original image (height, width)
        tile_size: size of each tile
        tiles_in_x: number of tiles in the x direction
        tiles_in_y: number of tiles in the y direction
    Returns:
        A single prediction batch of recombined tiles.
    """
    original_height, original_width = original_size
    b, c, _, _ = tiles[0].shape

    # Create tensors for the recombined prediction and the count for averaging
    recombined = torch.zeros((b, c, original_height, original_width), dtype=tiles[0].dtype, device=tiles[0].device)
    count_matrix = torch.zeros_like(recombined)

    tile_index = 0
    for i in range(tiles_in_y):
        for j in range(tiles_in_x):
            # Calculate start coordinates for the tile
            start_x = j * (tile_size - (tile_size * tiles_in_x - original_width) // (tiles_in_x - 1))
            start_y = i * (tile_size - (tile_size * tiles_in_y - original_height) // (tiles_in_y - 1))
            end_x = min(start_x + tile_size, original_width)
            end_y = min(start_y + tile_size, original_height)

            # Adjust the size of the tile if it's an edge tile
            tile = tiles[tile_index]
            if tile.shape[2] != end_y - start_y or tile.shape[3] != end_x - start_x:
                tile = tile[:, :, :end_y - start_y, :end_x - start_x]

            # Place tile and update count matrix
            recombined[:, :, start_y:end_y, start_x:end_x] += tile
            count_matrix[:, :, start_y:end_y, start_x:end_x] += 1

            tile_index += 1

    # Avoid division by zero
    count_matrix[count_matrix == 0] = 1

    # Average the overlapped regions
    recombined /= count_matrix

    return recombined

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res

def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res

def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res

def one_hot2hd_dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                    dtype=None) -> np.ndarray:
    """
    Used for https://arxiv.org/pdf/1904.10030.pdf,
    implementation from https://github.com/JunMa11/SegWithDistMap
    """
    # Relasx the assertion to allow computation live on only a
    # subset of the classes
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            res[k] = eucl_distance(posmask, sampling=resolution)

    return res

def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res

def test_fn():
    print('This is a test function')


def inference_fn(model, img:torch.Tensor, 
                 tta_lambdas='flips', device='cuda',
                 tile_size=TILE_SIZE,
                 noise_amount=NOISE_MULTIPLIER, pred_threshold=PREDICTION_THRESHOLD):
    # add noise to the image
    noise = torch.randn_like(img) * noise_amount
    img = img + noise
    
    img = img.to(device)
    model = model.to(device)
    
    b, c, h, w = img.shape
    
    tiles_in_x, tiles_in_y = get_tile_nums(h, w, tile_size, min_overlap=0.5)
    
    preds = torch.zeros((b, 1, h, w), dtype=torch.float32, device=device)
    
    if tta_lambdas == 'flips':
        tta_lambdas = [lambda x: x, 
                       lambda x: torch.flip(x, dims=[3]),
                       lambda x: torch.flip(x, dims=[2]), 
                       lambda x: torch.flip(torch.flip(x, dims=[2]), dims=[3])]
    else:
        assert type(tta_lambdas[0]) == type(lambda x: x), 'tta_lambdas must be a list of functions'
        tta_lambdas = tta_lambdas
    
    for tta_fn in tta_lambdas:
        tta_img = tta_fn(img)
        tiles = batch_tiling_split(tta_img, tile_size, tiles_in_x=tiles_in_x, tiles_in_y=tiles_in_y)
        
        model.eval()
        with torch.no_grad():
            tile_preds = [model(tile) for tile in tiles]
            
        tta_preds = recombine_tiles(tile_preds, (h, w), TILE_SIZE, tiles_in_x=tiles_in_x, tiles_in_y=tiles_in_y)
        preds += tta_fn(tta_preds)
            
    preds /= len(tta_lambdas)

    preds = (nn.Sigmoid()(preds)>pred_threshold).double()
    preds = preds.cpu().numpy().astype(np.uint8)
    return preds


def plot_examples(model, num=5, device='cuda', 
                  dataset_folder=VAL_DATASET_DIR, sub_data_idxs=None,
                  save=False, save_dir=None, show=True):
    
    _img_path = os.path.join(dataset_folder, 'images', "*"+IMG_FILE_EXT)
    img_paths = sorted(glob(os.path.join(dataset_folder, 'images', "*"+IMG_FILE_EXT)))
    mask_paths = sorted(glob(os.path.join(dataset_folder, 'labels', "*"+MASK_FILE_EXT)))
    
    assert len(img_paths) == len(mask_paths), 'Number of images and masks must be equal'

    if sub_data_idxs is not None:
        start, end = sub_data_idxs
        img_paths = img_paths[start:end]
        mask_paths = mask_paths[start:end]
    
    # print(f'Looking in {_img_path} found {len(img_paths)} images')
    # choose n random images
    rand_idxs = np.random.choice(len(img_paths), num, replace=False)

    h, w, c = cv2.imread(img_paths[0]).shape

    plt.figure(figsize=(3*w/600, num*h/600))
    subplot_idx = 1
    for i, idx in enumerate(rand_idxs):
        img_path = img_paths[idx]
        mask_path = mask_paths[idx]
        
        img = preprocess_image(img_path)
        msk = preprocess_mask(mask_path)
        
        pred = inference_fn(model, img.unsqueeze(0).to(device))
        
        if c > 1:
            img = img[0] # only plot one channel
        
        plt.subplot(num, 3, subplot_idx)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title('Original Image') if i == 0 else None # only add title to first row
        plt.subplot(num, 3, subplot_idx+1)
        plt.imshow(msk.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title('Ground Truth') if i == 0 else None
        plt.subplot(num, 3, subplot_idx+2)
        plt.imshow(pred.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title('Prediction') if i == 0 else None
        subplot_idx += 3
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # add some space at the top of the figure for the title
    
    if save:
        save_dir = save_dir if save_dir is not None else 'saved_images/plot_examples.png'
        plt.savefig(save_dir)
    if show:
        plt.show()
    else:
        plt.close()

def show_image_pred(img, pred, mask=None, 
                    figsize=(10, 6),
                    title='Image, Prediction, (Mask)',
                    save=False, save_dir=None, show=True):
    """
    Plots the original image, the prediction and the mask side by side
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].imshow(img[0], cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(pred.squeeze(), cmap='gray')
    axs[1].set_title('Prediction')
    axs[1].axis('off')
    if mask is not None:
        axs[2].imshow(mask.squeeze(), cmap='gray')
        axs[2].set_title('Mask')
        axs[2].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # add some space at the top of the figure for the title
    
    if save:
        save_dir = save_dir if save_dir is not None else 'saved_images/plot_examples.png'
        plt.savefig(save_dir)
    if show:
        plt.show()
    else:
        plt.close()

def get_tile_nums(h, w, tile_size=TILE_SIZE, min_overlap=0.5):
    """
    Returns the number of tiles in the x and y direction for the given image
    """
    tiles_in_x = 1 if w == tile_size else math.ceil(w / (tile_size * (1 - min_overlap)))
    tiles_in_y = 1 if h == tile_size else math.ceil(h / (tile_size * (1 - min_overlap)))
    return tiles_in_x, tiles_in_y# wtf

def remove_small_objects(img, min_size):
    # Find all connected components (labels)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Create a mask where small objects are removed
    new_img = np.zeros_like(img)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            new_img[labels == label] = 1

    return new_img

def local_surface_dice_multiscale(models, sizes, device, dataset_folder="/data/train/kidney_2", sub_data_idxs=None, verbose=False):
    
    ls_images = glob(os.path.join(dataset_folder, "images", "*"+IMG_FILE_EXT))
    ls_masks = glob(os.path.join(dataset_folder, 'labels', "*"+MASK_FILE_EXT))
    if sub_data_idxs:
        ls_images = ls_images[sub_data_idxs[0]:sub_data_idxs[1]]
        ls_masks = ls_masks[sub_data_idxs[0]:sub_data_idxs[1]]
    
    # test_loader = create_test_loader(ls_images, BATCH_SIZE, augmentations=None)
    val_loader = create_loader(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, 
                               transform=val_transform, shuffle=False, 
                               sub_data_idxs=sub_data_idxs)

    # create a 3d tensor of zeros to store the predictions for the whole kidney for 3D surface dice score
    # also need a 3d tensor of zeros for the true mask
    h, w = cv2.imread(ls_images[0]).shape[:2]
    three_d_preds = torch.zeros((len(ls_images), h, w))
    true_masks = torch.zeros((len(ls_images), h, w))
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validating ') if verbose else enumerate(val_loader)
    for batch_idx, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        
        b, c, h, w = images.shape
        
        if batch_idx == 0 and verbose:
            print(f'Original shape: b={b}, c={c}, h x w = {h}x{w}')
        
        # TTA
        tta_lambdas = [lambda x: x, 
                       lambda x: torch.flip(x, dims=[3]),
                       lambda x: torch.flip(x, dims=[2]), 
                       lambda x: torch.flip(torch.flip(x, dims=[2]), dims=[3])]    
        
        predictions = []
        
        for i in range(len(sizes)):
            preds = torch.zeros((b, 1, h, w)).to(device)
            model = models[i]
            size = sizes[i]
            tiles_in_x, tiles_in_y = get_tile_nums(h, w, tile_size=size)    
            for tta_fn in tta_lambdas:
                tta_img = tta_fn(images)
                tiles = batch_tiling_split(tta_img, size, tiles_in_x=tiles_in_x, tiles_in_y=tiles_in_y)
                noise = torch.randn_like(tiles[0])*NOISE_MULTIPLIER
                
                model.eval()
                with torch.no_grad():
                    tile_preds = [model(tile+torch.randn_like(tile)*NOISE_MULTIPLIER) for tile in tiles]
                
                tta_preds = recombine_tiles(tile_preds, (h, w), size, tiles_in_x=tiles_in_x, tiles_in_y=tiles_in_y)
                # if verbose and batch_idx == 0:
                #     show_image_pred(tta_img[0].cpu(), F.sigmoid(tta_preds[0]).cpu(), mask=tta_fn(masks.unsqueeze(1))[0][0].cpu())
                # preds += torch.flip(tta_fn(tta_preds), dims=[3])
                # s_pred = recombine_tiles(s_tile_preds, (s_h, s_w), TILE_SIZE, tiles_in_x=TILES_IN_X, tiles_in_y=TILES_IN_Y)
                # s_pred = tta_fn(s_pred)
                # preds += F.interpolate(s_pred, size=(h, w), mode='bilinear', align_corners=False)
                
                preds += tta_fn(tta_preds)
                
            preds /= len(tta_lambdas)
            preds = torch.sigmoid(preds) # make sure the predictions are between 0 and 1
            preds = (preds>PREDICTION_THRESHOLD).double() # threshold the predictions
            predictions.append(preds)
        
        # flattern the predictions keeping the union of the predictions, all pixels that are 1 in any of the predictions
        preds = torch.zeros_like(predictions[0])
        for pred in predictions:
            preds += pred
        preds = (preds>0).double()
        preds = preds.cpu().numpy().astype(np.uint8)
        
        for i, pred in enumerate(preds):
            # true_mask = preprocess_mask(ls_masks[batch_idx*BATCH_SIZE+i])

            # The dimensions of the true mask and pred are [width, height] but need to be [1, width, height]
            # they need to stay as numpy arrays not torch tensors
            # pred = np.expand_dims(pred, axis=0)
            # true_mask = np.expand_dims(true_mask, axis=0)
            # true_mask = true_mask.squeeze()
            true_mask = masks[i].squeeze().cpu()
            pred = pred.squeeze()
            # pred = remove_small_objects(pred, 100)

            if verbose and batch_idx == 0 and i == 0:
                print(f'pred shape: {pred.shape}, true_mask shape: {true_mask.shape}')
                # plot an example
                # pred = pred[0]
                print(f'Prediction shape: {pred.shape}')
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(images[0].cpu().permute(1, 2, 0), cmap='gray')
                plt.title('image')
                plt.subplot(1, 2, 2)
                plt.imshow(pred)
                plt.title('prediction')
                plt.show()
                # pred = np.expand_dims(pred, axis=0)

            # print(f'pred shape: {pred.shape}, true_mask shape: {true_mask.shape}')
            three_d_preds[batch_idx*BATCH_SIZE+i] = torch.tensor(pred)
            true_masks[batch_idx*BATCH_SIZE+i] = true_mask.clone().detach()
            # print(f'3d preds shape: {three_d_preds.shape}, true_masks shape: {true_masks.shape}')
            
    three_d_preds = three_d_preds.numpy()
    true_masks = true_masks.numpy()
    surface_dice_3d = fast_compute_surface_dice_score_from_tensor(three_d_preds, true_masks)
    return surface_dice_3d#mean_surface_dice_score


def plot_list(images, n_cols=4):
    """
    Plots a list of images
    """
    n_rows = math.ceil(len(images)/n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 10*n_rows))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.show()
