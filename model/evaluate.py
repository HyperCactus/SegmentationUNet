"""
Loads the model and runs it on the test set. Saves the predictions to a folder for visualization.
and calculates the dice score on the test set.
"""

from dataset import *
import torch
from utils import *
from modules import ImprovedUNet
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from global_params import *

# set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)
    
    # calculate the dice score on the test set
    dice_score = calc_dice_score(model, VAL_LOADER, device=device, verbose=True)
    # dice_score = evaluate(model, VAL_LOADER, device=device, verbose=True, score_type='dice')
    # dice_score = eval_and_plot(model, VAL_LOADER, device=device, verbose=True)
    # dice_score, iou_score = evaluate(model, VAL_LOADER, device=device, verbose=True, score_type='both')
    print(f'Validation Dice Score: {dice_score:.4f}')
    # print(f'Validation IoU Score: {iou_score:.4f}')
    
    save_predictions_as_imgs(VAL_LOADER, model, num=30, folder='saved_images/', device=device)

    plot_samples(6, title='Predictions', include_image=True)

if __name__ == '__main__':
    main()