"""
This file is the main training script for the Improved UNet model
"""
# Copied from COMP3710 report

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import ImprovedUNet
from dataset import CustomDataset, augment_image
from sklearn.model_selection import train_test_split
import time
from utils import *
from costom_loss import FocalLoss, EpicLoss, BlackToWhiteRatioLoss, IoULoss
from global_params import * # Hyperparameters and other global variables
# from dice_loss import BinaryDiceLoss

# set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





LOAD_MODEL = True
SAVE_EPOCH_DATA = False#True

def train_epoch(loader, model, optimizer, loss_fn, scaler, losses):
    """Trains the model for one epoch

    Args:
        loader (DataLoader): The data loader
        model (nn.Module): The model
        optimizer (torch.optim): The optimizer
        loss_fn (nn.Module): The loss function
        scaler (torch.cuda.amp.GradScaler): The gradient scaler
        losses (list): The list to store the losses
    """
    eval = BinaryDiceScore()
    loop = tqdm(loader)
    length = len(loader)
    # loop = loader
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        
        # print(f'type(loss): {type(loss)}, shape: {loss.shape}')
        # print(f'loss: {loss.item()}')
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    images_path = os.path.join(base_path, dataset, 'images')
    labels_path = os.path.join(base_path, dataset, 'labels')

    image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.tif')])
    label_files = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.tif')])

    train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
        image_files, label_files, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_image_files, train_mask_files, augmentation_transforms=augment_image)
    val_dataset = CustomDataset(val_image_files, val_mask_files, augmentation_transforms=augment_image)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3 channels in for RGB images, 1 channel out for binary mask
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)
    
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = FocalLoss(gamma=2) # Focal Loss dosen't seem to be working, try changing output layer
    loss_fn = EpicLoss() # Custom loss
    # loss_fn = IoULoss() # Testing this loss function
    # loss_fn = BlackToWhiteRatioLoss() # Testing this loss function
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam optimizer
    # This learning rate scheduler reduces the learning rate by a factor of 0.1 if the mean epoch loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.1)
    
    # load model if LOAD_MODEL is True
    if LOAD_MODEL:
        load_checkpoint(torch.load('checkpoints/checkpoint.pth.tar'), model, optimizer)
    
    scaler = torch.cuda.amp.GradScaler()
    
    losses = [] # all training losses
    dice_scores = [] # for plotting
    epoch_losses = [] # average loss for each epoch
    
    model.train()
    
    # Training loop
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train_epoch_losses = [] # train losses for the epoch
        # Train the model for one epoch
        train_epoch(train_loader, model, optimizer, loss_fn, scaler, train_epoch_losses)
        
        # Calculate the average loss for the epoch
        average_loss = np.mean(train_epoch_losses)
        epoch_losses.append(average_loss)
        losses.extend(train_epoch_losses)
        
        # Update the learning rate
        scheduler.step(epoch_losses[-1])
        
        # Calculate the validation dice score after each epoch
        val_dice_score = evaluate(model, val_loader, device=device, verbose=True, leave_on_train=True)
        val_dice_score = np.round(val_dice_score.item(), 4)
        dice_scores.append(val_dice_score)
        print(f'Validation dice score: {val_dice_score}')
        print(f'Average epoch loss: {average_loss:.4f}')
            
        # Print some feedback after each epoch
        print_progress(start_time, epoch, NUM_EPOCHS)
        
        if SAVE_EPOCH_DATA:
            # Save some predictions to a folder for visualization
            os.makedirs(f'epoch_data', exist_ok=True)
            os.makedirs(f'epoch_data/epoch_{epoch}', exist_ok=True)
            os.makedirs(f'epoch_data/epoch_{epoch}/checkpoints', exist_ok=True)
            os.makedirs(f'epoch_data/epoch_{epoch}/images', exist_ok=True)
            save_predictions_as_imgs(val_loader, model, 10, folder=f'epoch_data/epoch_{epoch}/images/', device=device)
            # Save a checkpoint after each epoch
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=f'epoch_data/epoch_{epoch}/checkpoints/checkpoint.pth.tar')
        
    # Save a checkpoint after training is complete
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    save_checkpoint(checkpoint)
        
    # Plot the losses
    plt.figure(figsize=(20, 10))
    plt.plot(losses, label='Loss')
    plt.xlabel('Batch #')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.grid(True)
    plt.savefig('save_data/losses.png')
    plt.show()
    
    # plot Average Losses per Epoch
    plt.figure(figsize=(20, 10))
    plt.plot(epoch_losses, label='Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.title('Average Losses per Epoch')
    plt.grid(True)
    plt.savefig('save_data/epoch_losses.png')
    plt.show()
    
    # plot dice score vs epoch
    plt.figure(figsize=(20, 10))
    plt.plot(dice_scores, label='Dice Score')
    plt.xlabel('Epoch #')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Scores')
    plt.grid(True)
    plt.savefig('save_data/dice_scores.png')
    plt.show()

if __name__ == '__main__':
    main()