"""
This file is the main training script for the Improved UNet model
"""
# Copied from COMP3710 report

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import ImprovedUNet
from dataset import VAL_LOADER, TRAIN_LOADER, create_loader, augment_image # TRAIN_TRANSFORMS
import time
from utils import *
from torch.utils.tensorboard import SummaryWriter
from costom_loss import FocalLoss, EpicLoss, BlackToWhiteRatioLoss, IoULoss, ReduceLROnThreshold
from global_params import * # Hyperparameters and other global variables
from evaluate import surface_dice

# RANGPUR Settings 
from evaluate import main as evaluate_fn

# from dice_loss import BinaryDiceLoss

# set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for HPC
if not torch.cuda.is_available():
    # stop the script if GPU is not available
    print('GPU not available')
    exit()


LOAD_MODEL = False#True
SAVE_EPOCH_DATA = False#True
check_memory = True

writer = SummaryWriter('runs/SenNet/VascularSegmentation')
STEP = 0

def train_epoch(loader, model, optimizer, loss_fn, scaler, losses, 
                accuracies=None, check_memory=check_memory, variances=[]):
    """Trains the model for one epoch

    Args:
        loader (DataLoader): The data loader
        model (nn.Module): The model
        optimizer (torch.optim): The optimizer
        loss_fn (nn.Module): The loss function
        scaler (torch.cuda.amp.GradScaler): The gradient scaler
        losses (list): The list to store the losses
        accuracies (list): The train accuracies for plotting
    """
    # eval = BinaryDiceScore()
    loop = tqdm(loader)
    length = len(loader)
    # loop = loader
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        # targets = targets.float().unsqueeze(1).to(device=device)
        targets = targets.float().to(device=device)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # print(f'Data shape: {data.shape}\n Targets shape: {targets.shape}\n Preds shape: {predictions.shape}')
            # print(f'Median: {torch.median(predictions)}')
            # sigmoid
            
            # predictions shape [batch size, 1, 512, 512], target shape [batch size, 512, 512] must be same
            # squees out dim 1, so predictions shape [batch size, 512, 512]
            predictions = torch.squeeze(predictions, dim=1)
            loss = loss_fn(predictions, targets)
            if loss.isnan():
        # backward
        optimizer.zero_grad()

        if check_memory and batch_idx == 0:
            t = torch.cuda.get_device_properties(0).total_memory / 1024**3
            a = torch.cuda.memory_allocated(0) / 1024**3
            print(f'MEMORY USAGE: {a:.2f}GB out of {t:.2f}GB ({a/t*100:.2f}%)')
            check_memory = False

        # if batch_idx % 10 == 0:
        #     preds = (nn.Sigmoid()(predictions)>PREDICTION_THRESHOLD).double()
        
        # print(f'type(loss): {type(loss)}, shape: {loss.shape}')
        # print(f'loss: {loss.item()}')
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        var = np.var(losses)
        # stabillity = 1 / (1 + var)
        global STEP
        writer.add_scalar('train_batch_loss', loss.item(), STEP)
        STEP += 1
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item(), var=var)

def train():    
    # 3 channels in for RGB images, 1 channel out for binary mask
    model = ImprovedUNet(in_channels=IN_CHANNELS, out_channels=1).to(device)
    
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = FocalLoss(gamma=2) # Focal Loss dosen't seem to be working, try changing output layer
    loss_fn = EpicLoss() # Custom loss
    # loss_fn = IoULoss() # Testing this loss function
    # loss_fn = BlackToWhiteRatioLoss() # Testing this loss function
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam optimizer
    # This learning rate scheduler reduces the learning rate by a factor of 0.1 if the mean epoch loss plateaus
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.1)
    scheduler = ReduceLROnThreshold(optimizer, threshold=0.001, mode='above', verbose=True, factor=0.1)
    
    # load model if LOAD_MODEL is True
    if LOAD_MODEL:
        load_checkpoint(torch.load('checkpoints/checkpoint.pth.tar'), model, optimizer)
    
    scaler = torch.cuda.amp.GradScaler()
    
    losses = [] # all training losses
    dice_scores = [] # for plotting
    epoch_losses = [] # average loss for each epoch
    train_surface_dice_scores = [] # for plotting
    epoch_variances = [] # average loss variance for each epoch
    
    model.train()
    check_memory = True
    # Training loop
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train_epoch_losses = [] # train losses for the epoch
        train_epoch_variances = [] # train loss variance for the epoch
        # Train the model for one epoch
        train_epoch(TRAIN_LOADER, model, optimizer, loss_fn, scaler, 
                    train_epoch_losses, check_memory=check_memory, variances=train_epoch_variances)
        check_memory = False
        
        # Calculate the average loss for the epoch
        average_loss = np.mean(train_epoch_losses)
        average_loss_variance = np.mean(train_epoch_variances)
        epoch_loss_variance = np.var(train_epoch_losses)
        epoch_variances.append(np.mean(epoch_loss_variance))
        epoch_losses.append(average_loss)
        losses.extend(train_epoch_losses)
        
        # Update the learning rate
        # scheduler.step(epoch_losses[-1])
        scheduler.step(average_loss_variance)
        
        # Calculate the validation dice score after each epoch
        # val_dice_score = evaluate(model, VAL_LOADER, device=device, verbose=True, leave_on_train=True)
        val_dice_score = surface_dice(model, device=device, loader=VAL_LOADER, data_dir=VAL_DATASET_DIR)
        val_dice_score = np.round(val_dice_score.item(), 4)
        dice_scores.append(val_dice_score)
        print(f'Validation dice score: {val_dice_score}')
        print(f'Average epoch loss: {average_loss:.4f}')
        print(f'Epoch loss variance: {epoch_loss_variance:.4f}')
            
        # Print some feedback after each epoch
        print_progress(start_time, epoch, NUM_EPOCHS)
        
        if SAVE_EPOCH_DATA:
            # Save some predictions to a folder for visualization
            os.makedirs(f'epoch_data', exist_ok=True)
            os.makedirs(f'epoch_data/epoch_{epoch}', exist_ok=True)
            os.makedirs(f'epoch_data/epoch_{epoch}/checkpoints', exist_ok=True)
            os.makedirs(f'epoch_data/epoch_{epoch}/images', exist_ok=True)
            save_predictions_as_imgs(VAL_LOADER, model, 10, folder=f'epoch_data/epoch_{epoch}/images/', device=device)
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
    
    # evaluate_fn()
        
    # Plot the losses
    plt.figure(figsize=(20, 10))
    plt.plot(losses, label='Loss')
    plt.xlabel('Batch #')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.grid(True)
    plt.savefig('save_data/losses.png')
    # plt.show()
    
    # plot Average Losses per Epoch
    plt.figure(figsize=(20, 10))
    plt.plot(epoch_losses, label='Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.title('Average Losses per Epoch')
    plt.grid(True)
    plt.savefig('save_data/epoch_losses.png')
    # plt.show()
    
    # plot dice score vs epoch
    plt.figure(figsize=(20, 10))
    plt.plot(dice_scores, label='Surface Dice Score')
    plt.xlabel('Epoch #')
    plt.ylabel('Surface Dice Score')
    plt.title('Validation Surface Dice Scores')
    plt.grid(True)
    plt.savefig('save_data/dice_scores.png')
    # plt.show()

    # plot loss variance vs epoch
    plt.figure(figsize=(20, 10))
    plt.plot(epoch_variances, label='Loss Variance')
    plt.xlabel('Epoch #')
    plt.ylabel('Train Loss Variance')
    plt.title('Average Loss Variance per Epoch')
    plt.grid(True)
    plt.savefig('save_data/epoch_variances.png')
    # plt.show()

    print('TRAIN COMPLETE')

if __name__ == '__main__':
    train()