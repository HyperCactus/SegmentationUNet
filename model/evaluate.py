"""
Loads the model and runs it on the test set. Saves the predictions to a folder for visualization.
and calculates the dice score on the test set.
"""

from dataset import CustomDataset, augment_image, val_augment_image
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

BATCH_SIZE = 1 # We want to test one image at a time
NUM_WORKERS = 1
CHECKPOINT_DIR = 'checkpoints/checkpoint.pth.tar'

def main():
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)

    images_path = os.path.join(base_path, dataset, 'images')
    labels_path = os.path.join(base_path, dataset, 'labels')

    image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.tif')])
    label_files = sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.tif')])

    _, val_image_files, _, val_mask_files = train_test_split(image_files, label_files, test_size=0.2, random_state=42)
        
    # train_dataset = CustomDataset(train_image_files, train_mask_files, augmentation_transforms=augment_image)
    val_dataset = CustomDataset(val_image_files, val_mask_files, augmentation_transforms=val_augment_image)

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # calculate the dice score on the test set
    # dice_score = calc_dice_score(model, val_loader, device=device, verbose=True)
    dice_score = eval_and_plot(model, val_loader, device=device, verbose=True)
    # dice_score = evaluate(model, val_loader, device=device, verbose=True)
    print(f'Test Dice Score: {dice_score:.4f}')
    
    save_predictions_as_imgs(val_loader, model, num=30, folder='saved_images/', device=device)

    plot_samples(6, title='Predictions', include_image=True)

if __name__ == '__main__':
    main()