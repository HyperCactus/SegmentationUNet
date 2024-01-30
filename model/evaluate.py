"""
Loads the model and runs it on the test set. Saves the predictions to a folder for visualization.
and calculates the dice score on the test set.
"""

from dataset import UsageDataset, VAL_LOADER, val_transform, create_loader
from volumetric_dataset2 import create_loader as create_volumetric_loader
import torch
from utils import *
from modules import ImprovedUNet
from torch.utils.data import DataLoader
# from IPython.display import display
# from surface_dice import score
import albumentations as A
from glob import glob
from albumentations.pytorch import ToTensorV2
from helper import fast_compute_surface_dice_score_from_tensor
from surface_dice import compute_surface_dice_score, add_size_columns
from global_params import *
import torch.nn.functional as F


def local_surface_dice(model, device, dataset_folder="/data/train/kidney_2", 
                       sub_data_idxs=None, verbose=False, fast_mode=False):
    
    ls_images = glob(os.path.join(dataset_folder, "images", "*"+IMG_FILE_EXT))
    ls_masks = glob(os.path.join(dataset_folder, 'labels', "*"+MASK_FILE_EXT))
    if sub_data_idxs:
        ls_images = ls_images[sub_data_idxs[0]:sub_data_idxs[1]]
        ls_masks = ls_masks[sub_data_idxs[0]:sub_data_idxs[1]]
    
    # test_loader = create_test_loader(ls_images, BATCH_SIZE, augmentations=None)
    if fast_mode:
        val_loader = create_volumetric_loader(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, 
                               shuffle=False, subvol_start_pos='center')
    else:
        val_loader = create_loader(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, 
                                transform=val_transform, shuffle=False, 
                                sub_data_idxs=sub_data_idxs)

    # create a 3d tensor of zeros to store the predictions for the whole kidney for 3D surface dice score
    # also need a 3d tensor of zeros for the true mask
    h, w = cv2.imread(ls_images[0]).shape[:2]
    if fast_mode:
        # there are 3 axes in the prediction, every 3rd prediction is a new slice along an axis
        z_preds_vol = torch.zeros((TILE_SIZE, TILE_SIZE, TILE_SIZE))
        x_preds_vol = torch.zeros((TILE_SIZE, TILE_SIZE, TILE_SIZE))
        y_preds_vol = torch.zeros((TILE_SIZE, TILE_SIZE, TILE_SIZE))
        mask_vol = torch.zeros((TILE_SIZE, TILE_SIZE, TILE_SIZE))
    
    pred_rles = []
    mask_rels = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validating ')
    for batch_idx, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)        
        preds = inference_fn(model, images) # predict
        
        for i, pred in enumerate(preds):
            true_mask = masks[i].squeeze().cpu()
            pred = pred.squeeze()

            if verbose and batch_idx == 0 and i == 0:
                print(f'pred shape: {pred.shape}, true_mask shape: {true_mask.shape}')
                # plot an example
                print(f'Prediction shape: {pred.shape}')
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 3, 1)
                plt.imshow(images[0].cpu().permute(1, 2, 0), cmap='gray')
                plt.title('image')
                plt.subplot(1, 3, 2)
                plt.imshow(true_mask, cmap='gray')
                plt.title('true mask')
                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap='gray')
                plt.title('prediction')
                
                plt.show()
                # pred = np.expand_dims(pred, axis=0)

            # print(f'pred shape: {pred.shape}, true_mask shape: {true_mask.shape}')
            # three_d_preds[batch_idx*BATCH_SIZE+i] = pred#.clone().detach()
            # true_masks[batch_idx*BATCH_SIZE+i] = true_mask#.clone().detach()
            # print(f'3d preds shape: {three_d_preds.shape}, true_masks shape: {true_masks.shape}')
            if fast_mode:
                # every 3rd prediction is a new slice along an axis
                if batch_idx*BATCH_SIZE+i % 3 == 0: # slice with contains z-axis
                    z_preds_vol[:, :, (batch_idx*BATCH_SIZE+i)//3] = pred
                    mask_vol[:, :, (batch_idx*BATCH_SIZE+i)//3] = true_mask
                elif batch_idx*BATCH_SIZE+i % 3 == 1: # slice with contains x-axis
                    x_preds_vol[:, (batch_idx*BATCH_SIZE+i-1//3), :] = pred
                else: # slice with contains y-axis
                    y_preds_vol[(batch_idx*BATCH_SIZE+i-2)//3, :, :] = pred
                    
            else:
                rle = rle_encode(pred)
                if rle == '':
                    rle = '1 0'
                pred_rles.append(rle)
                mask_rel = rle_encode(true_mask)
                if mask_rel == '':
                    mask_rel = '1 0'
                mask_rels.append(mask_rel)
    
    if fast_mode:
        # sanity check
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(z_preds_vol[:, :, 0], cmap='gray')
        plt.title('z preds')
        plt.subplot(1, 3, 2)
        plt.imshow(x_preds_vol[0, :, :], cmap='gray')
        plt.title('x preds')
        plt.subplot(1, 3, 3)
        plt.imshow(y_preds_vol[:, 0, :], cmap='gray')
        # rotate the x and y preds to match the z preds
        x_preds_vol = x_preds_vol.permute(2, 1, 0)
        y_preds_vol = y_preds_vol.permute(0, 2, 1)
        # plot first slice of each volume to check
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(z_preds_vol[:, :, 0], cmap='gray')
        plt.title('z preds')
        plt.subplot(1, 3, 2)
        plt.imshow(x_preds_vol[:, :, 0], cmap='gray')
        plt.title('x preds')
        plt.subplot(1, 3, 3)
        plt.imshow(y_preds_vol[:, :, 0], cmap='gray')
        plt.title('y preds')
        plt.show()
        
        # average the prediction tensors together
        three_d_preds = (z_preds_vol + x_preds_vol + y_preds_vol) / 3
        
        surface_dice_3d = fast_compute_surface_dice_score_from_tensor(three_d_preds.numpy(), mask_vol.numpy())
    else:
        ids = []
        for p_img in tqdm(ls_images):
            path_ = p_img.split(os.path.sep)
            # parse the submission ID
            dataset = path_[-3].split('/')[-1]
            slice_id, _ = os.path.splitext(path_[-1])
            ids.append(f"{dataset}_{slice_id}")
        
        preds_df = pd.DataFrame.from_dict({
            "id": ids,
            "rle": pred_rles
        })
        masks_df = pd.DataFrame.from_dict({
            "id": ids,
            "rle": mask_rels
        })
        add_size_columns(masks_df)
        
        surface_dice_3d = compute_surface_dice_score(preds_df, masks_df)
            
    # three_d_preds = three_d_preds.numpy()
    # true_masks = true_masks.numpy()
    # surface_dice_3d = fast_compute_surface_dice_score_from_tensor(three_d_preds, true_masks)
    return surface_dice_3d#mean_surface_dice_score

def main():
    # set the device to cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_512 = ImprovedUNet(in_channels=IN_CHANNELS, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model_512)

    surface_dice_score = local_surface_dice(model_512, device, dataset_folder=VAL_DATASET_DIR, 
                                            verbose=True, fast_mode=True)

    print(f'Surface Dice Score: {surface_dice_score:.4f}')
    
    plot_examples(model_512)

if __name__ == '__main__':
    main()