"""
This file is the main training script for the Improved UNet model
"""
# Copied from COMP3710 report

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from modules import ImprovedUNet
from dataset import VAL_LOADER, TRAIN_LOADER, create_loader, augment_image, \
    preprocess_image, preprocess_mask # TRAIN_TRANSFORMS
import time
from utils import *
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from costom_loss import CustomFocalLoss, EpicLoss, BlackToWhiteRatioLoss, IoULoss, ReduceLROnThreshold, \
    BinaryDiceLoss, BoundaryDoULoss, IoUDiceLoss
from global_params import * # Hyperparameters and other global variables
from evaluate import local_surface_dice as validate
from PIL import Image
import cv2

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
if TEST_MODE:
    # overfit model on a single batch
    # TRAIN_LOADER = [next(iter(TRAIN_LOADER))]
    img_path = 'data_png/train/kidney_1_dense/images/0996.png'
    msk_path = 'data_png/train/kidney_1_dense/labels/0996.png'
    img = preprocess_image(img_path)
    msk = preprocess_mask(msk_path)
    # convert to tensor
    img = torch.tensor(img).float()
    msk = torch.tensor(msk).float()
    # resize to IMAGE_HEIGHT, IMAGE_WIDTH
    img = F.interpolate(img.unsqueeze(0), size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear', align_corners=False)
    msk = F.interpolate(msk.unsqueeze(0).unsqueeze(0), size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear', align_corners=False).squeeze(0)
    TRAIN_LOADER = [(img, msk)]

writer = SummaryWriter('runs/SenNet/VascularSegmentation')
writer.add_text('Hyperparameters', 
            f'Learning Rate: {LEARNING_RATE}, \
Batch Size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, Num Workers: {NUM_WORKERS}, \
Pin Memory: {PIN_MEMORY}, Prediction Threshold: {PREDICTION_THRESHOLD}, \
In Channels: {IN_CHANNELS}, Image Height: {IMAGE_HEIGHT}, Image Width: {IMAGE_WIDTH}, \
High Pass Alpha: {HIGH_PASS_ALPHA}, High Pass Strength: {HIGH_PASS_STRENGTH}, \
Tiles in X: {TILES_IN_X}, Tiles in Y: {TILES_IN_Y}, Tile Size: {TILE_SIZE}, \
Noise Multiplier: {NOISE_MULTIPLIER}', 0)
STEP = 0

def train_epoch(loader, model, optimizer, loss_fn, scaler, losses, 
                accuracies=None, check_memory=check_memory, variances=[], loop=None, epoch=0):
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
    this_loop = tqdm(loader) if loop is None else loader
    loop = this_loop if loop is None else loop
    loop.set_description('Training')
    length = len(loader)
    # loop = loader
    for batch_idx, (data, targets) in enumerate(this_loop):
        # this is for training to identify large ateries and veins
        if torch.sum(targets) > 10000 and np.random.random() > 0.5:
            continue
        
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)
        
        # if not TEST_MODE:
        noise = torch.randn_like(data) * NOISE_MULTIPLIER
        data += noise
        # forward
        with torch.cuda.amp.autocast():# and torch.autograd.detect_anomaly():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            if loss.isnan():
                print('loss is nan')
                print(f'predictions: {predictions}')
                print(f'targets: {targets}')
                # use exit code 3 to indicate that the model diverged
                exit(3)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if TEST_MODE and (epoch+1) % 5 == 0:
            msk = targets[0].cpu().numpy()
            img = data[0].cpu().numpy()
            pred = F.sigmoid(predictions[0]).cpu().detach().numpy()
            pred = (pred > PREDICTION_THRESHOLD).astype(np.uint8)
            save_dir=f'saved_images/testing/epoch_{epoch+1}.png'
            show_image_pred(img, pred, mask=msk, show=False, save=True, save_dir=save_dir)
            print('SAVED IMAGE')
            img = torch.tensor(cv2.imread(save_dir)).permute(2, 0, 1).float() / 255.0
            writer.add_image('example_predictions', img, epoch)

        if check_memory and batch_idx == 10:
            t = torch.cuda.get_device_properties(0).total_memory / 1024**3
            a = torch.cuda.memory_allocated(0) / 1024**3
            print(f'MEMORY USAGE: {a:.2f}GB out of {t:.2f}GB ({a/t*100:.2f}%)')
            check_memory = False
        
        losses.append(loss.item())
        var = np.var(losses)
        global STEP
        writer.add_scalar('train_batch_loss', loss.item(), STEP)
        STEP += 1
        loop.set_postfix(loss=loss.item()) # update tqdm loop

def train():
    begin_time = time.time()  
    # 3 channels in for RGB images, 1 channel out for binary mask
    # model = ImprovedUNet(in_channels=IN_CHANNELS, out_channels=1, features=[32,64,128,256,512]).to(device)
    model = ImprovedUNet(in_channels=IN_CHANNELS, out_channels=1).to(device)
    
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = BinaryDiceLoss()
    # loss_fn = FocalLoss(gamma=2) # Focal Loss dosen't seem to be working, try changing output layer
    # loss_fn = EpicLoss() # Custom loss
    # loss_fn = BoundaryDoULoss(1) # Testing this loss function
    # loss_fn = IoULoss(smooth=1) # Testing this loss function
    # loss_fn = BlackToWhiteRatioLoss() # Testing this loss function
    # loss_fn = IoUDiceLoss() # Testing this loss function
    loss_fn = CustomFocalLoss(alpha=0.1, gamma=5.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam optimizer
    # This learning rate scheduler reduces the learning rate by a factor of 0.1 if the mean epoch loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True, factor=0.1)
    # scheduler = ReduceLROnThreshold(optimizer, threshold=0.02, mode='above', verbose=True, factor=0.1)
    
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
    loop = tqdm(range(NUM_EPOCHS))
    for epoch in loop:
        train_epoch_losses = [] # train losses for the epoch
        train_epoch_variances = [] # train loss variance for the epoch
        # Train the model for one epoch
        train_epoch(TRAIN_LOADER, model, optimizer, loss_fn, scaler, 
                    train_epoch_losses, check_memory=check_memory, 
                    variances=train_epoch_variances, loop=loop, epoch=epoch)
        check_memory = False
        
        # Calculate the average loss for the epoch
        average_loss = np.mean(train_epoch_losses)
        # average_loss_variance = np.mean(train_epoch_variances)
        epoch_loss_variance = np.var(train_epoch_losses)
        epoch_variances.append(np.mean(epoch_loss_variance))
        epoch_losses.append(average_loss)
        losses.extend(train_epoch_losses)
        
        # Update the learning rate
        scheduler.step(epoch_losses[-1])
        # scheduler.step(epoch_variances[-1])
        if not TEST_MODE:
            plot_examples(model, num=5, device=device, 
                        dataset_folder=VAL_DATASET_DIR, 
                        sub_data_idxs=(500, 1400), save=True,
                        save_dir=f'saved_images/epoch_{epoch+1}_examples.png', show=False)
            # read the image with PIL and convert to numpy array
            img = torch.tensor(cv2.imread(f'saved_images/epoch_{epoch+1}_examples.png')).permute(2, 0, 1).float() / 255.0
            writer.add_image('example_predictions', img, epoch)
        
        # Calculate the validation dice score after each epoch
        # val_dice_score = evaluate(model, VAL_LOADER, device=device, verbose=True, leave_on_train=True)
        # select a random subvolume from the validation dataset
        n_images = len(glob(os.path.join(VAL_IMG_DIR, '*'+IMG_FILE_EXT)))
        
        if HPC:
            loop.set_description('Validating')
            subvol_depth = 500 if HPC else 1
            subvol_start = np.random.randint(0, n_images-subvol_depth)
            sub_data_idxs = (subvol_start, subvol_start+subvol_depth)
            val_dice_score = validate(model, device=device, dataset_folder=VAL_DATASET_DIR, 
                                    sub_data_idxs=sub_data_idxs, verbose=False)
            val_dice_score = np.round(val_dice_score, 4)
            loop.set_description(f'VAL SDC: {val_dice_score}')
            
            dice_scores.append(val_dice_score)
            writer.add_scalar('val_dice_score', val_dice_score, epoch)
            writer.add_scalar('epoch_loss', average_loss, epoch)
            print(f'Validation dice score: {val_dice_score}')
            print(f'Average epoch loss: {average_loss:.4f}')
            print(f'Epoch loss variance: {epoch_variances[-1]:.4f}')
            print(f'Actual Learning Rate: {optimizer.param_groups[0]["lr"]:.4e}')
        
        if epoch == NUM_EPOCHS-1 and HPC and NUM_EPOCHS > 8:
            # validate on full validation dataset
            ful_val_score = validate(model, device=device, dataset_folder=VAL_DATASET_DIR)
            print(f'Full validation 3D Surface Dice Score: {ful_val_score}')
            
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
    if not HPC:
        plt.show()
    
    # plot Average Losses per Epoch
    plt.figure(figsize=(20, 10))
    plt.plot(epoch_losses, label='Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.title('Average Losses per Epoch')
    plt.grid(True)
    plt.savefig('save_data/epoch_losses.png')
    if not HPC and not TEST_MODE:
        plt.show()
    
    # plot dice score vs epoch
    plt.figure(figsize=(20, 10))
    plt.plot(dice_scores, label='Surface Dice Score')
    plt.xlabel('Epoch #')
    plt.ylabel('Surface Dice Score')
    plt.title('Validation Surface Dice Scores')
    plt.grid(True)
    plt.savefig('save_data/dice_scores.png')
    if not HPC and not TEST_MODE:
        plt.show()

    # # plot loss variance vs epoch
    # plt.figure(figsize=(20, 10))
    # plt.plot(epoch_variances, label='Loss Variance')
    # plt.xlabel('Epoch #')
    # plt.ylabel('Train Loss Variance')
    # plt.title('Average Loss Variance per Epoch')
    # plt.grid(True)
    # plt.savefig('save_data/epoch_variances.png')
    # if not HPC and not TEST_MODE:
    #     plt.show()

    finish_time = time.time()
    # convert time to hours, minutes, seconds
    h, rem = divmod(finish_time-begin_time, 3600)
    m, s = divmod(rem, 60)
    print(f'TRAIN COMPLETE IN {h:.0f}h {m:.0f}m {s:.0f}s')

if __name__ == '__main__':
    train()