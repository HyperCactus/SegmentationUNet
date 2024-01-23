"""
Loads the model and runs it on the test set. Saves the predictions to a folder for visualization.
and calculates the dice score on the test set.
"""

from dataset import UsageDataset, VAL_LOADER, val_transform, create_loader
import torch
from utils import *
from modules import ImprovedUNet
from torch.utils.data import DataLoader
# from IPython.display import display
from surface_dice import score
import albumentations as A
from glob import glob
from albumentations.pytorch import ToTensorV2
from helper import fast_compute_surface_dice_score_from_tensor
from global_params import *


# set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_mask(path):
    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk = msk.astype('float32')
    msk/=255.0
    msk_ten = torch.tensor(msk)
    
    return msk_ten

def test_transform(image):
    
    image_np = image.permute(1, 2, 0).numpy()

    transform = A.Compose([
        # A.Resize(IMAGE_HEIGHT,IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
        A.Emboss(alpha=HIGH_PASS_ALPHA, strength=HIGH_PASS_STRENGTH, always_apply=True),  # High pass filter
    ])

    augmented = transform(image=image_np)
    augmented_image = augmented['image']

    augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)

    return augmented_image

def create_test_loader(image_files, batch_size, 
                  augmentations=None, shuffle=False):
    
    dataset = UsageDataset(image_files, augmentation_transforms=augmentations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def local_surface_dice(model, device, dataset_folder="/data/train/kidney_2", sub_data_idxs=None, verbose=False):
    
    ls_images = glob(os.path.join(dataset_folder, "images", "*"+IMG_FILE_EXT))
    ls_masks = glob(os.path.join(dataset_folder, 'labels', "*"+MASK_FILE_EXT))
    if sub_data_idxs:
        ls_images = ls_images[sub_data_idxs[0]:sub_data_idxs[1]]
        ls_masks = ls_masks[sub_data_idxs[0]:sub_data_idxs[1]]
    
    test_loader = create_test_loader(ls_images, BATCH_SIZE, augmentations=None)
    val_loader = create_loader(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, transform=None, shuffle=False, sub_data_idxs=sub_data_idxs)

    # create a 3d tensor of zeros to store the predictions for the whole kidney for 3D surface dice score
    # also need a 3d tensor of zeros for the true mask
    h, w = cv2.imread(ls_images[0]).shape[:2]
    three_d_preds = torch.zeros((len(ls_images), h, w))
    true_masks = torch.zeros((len(ls_images), h, w))
    
    surface_dice_scores = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validating ') if verbose else enumerate(val_loader)
    for batch_idx, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        
        orig_shape = images.shape[2:]
        
        if batch_idx == 0 and verbose:
            print(f'Original shape: {orig_shape}')
        
        # TTA
        tta_lambdas = [lambda x: x, 
                       lambda x: torch.flip(x, dims=[3]),
                       lambda x: torch.flip(x, dims=[2]), 
                       lambda x: torch.flip(torch.flip(x, dims=[2]), dims=[3])]    
            
        preds = torch.zeros((images.shape[0], 1, orig_shape[0], orig_shape[1])).to(device)
        
        for tta_fn in tta_lambdas:
            tta_img = tta_fn(images)
            tiles = batch_tiling_split(tta_img, TILE_SIZE, tiles_in_x=TILES_IN_X, tiles_in_y=TILES_IN_Y)
        
            model.eval()
            with torch.no_grad():
                tile_preds = [model(tile) for tile in tiles]
                
            tta_preds = recombine_tiles(tile_preds, orig_shape, TILE_SIZE, tiles_in_x=TILES_IN_X, tiles_in_y=TILES_IN_Y)
            # preds += torch.flip(tta_fn(tta_preds), dims=[3])
            preds += tta_fn(tta_preds)
                
        preds /= len(tta_lambdas)

        preds = (nn.Sigmoid()(preds)>PREDICTION_THRESHOLD).double()
        preds = preds.cpu().numpy().astype(np.uint8)

        for i, pred in enumerate(preds):
            # true_mask = preprocess_mask(ls_masks[batch_idx*BATCH_SIZE+i])

            # The dimensions of the true mask and pred are [width, height] but need to be [1, width, height]
            # they need to stay as numpy arrays not torch tensors
            # pred = np.expand_dims(pred, axis=0)
            # true_mask = np.expand_dims(true_mask, axis=0)
            # true_mask = true_mask.squeeze()
            true_mask = masks[i].squeeze()
            pred = pred.squeeze()
            
            # pred = remove_small_objects(pred, 5)

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
            true_masks[batch_idx*BATCH_SIZE+i] = torch.tensor(true_mask)
            # print(f'3d preds shape: {three_d_preds.shape}, true_masks shape: {true_masks.shape}')
            
    three_d_preds = three_d_preds.numpy()
    true_masks = true_masks.numpy()
    surface_dice_3d = fast_compute_surface_dice_score_from_tensor(three_d_preds, true_masks)
    return surface_dice_3d#mean_surface_dice_score

def main():
    model = ImprovedUNet(in_channels=IN_CHANNELS, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)

    surface_dice_score = local_surface_dice(model, device, dataset_folder=VAL_DATASET_DIR, 
                                            sub_data_idxs=(500, 515), verbose=True)

    print(f'Surface Dice Score: {surface_dice_score:.4f}')
    
    plot_examples(model, sub_data_idxs=(500, 1400))

if __name__ == '__main__':
    main()