"""
Loads the model and runs it on the test set. Saves the predictions to a folder for visualization.
and calculates the dice score on the test set.
"""

from dataset import *
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

def surface_dice(model, device=device, loader=VAL_LOADER, data_dir=f'{BASE_PATH}/kidney_2', sub_data_idxs=None):

    surface_dice_scores = []
    pbar = tqdm(enumerate(loader), total=len(loader), desc='Inference ')
    for batch_idx, (images, _transformed_masks) in pbar:
        if sub_data_idxs is not None and batch_idx*BATCH_SIZE < sub_data_idxs[0]:
            continue
        if sub_data_idxs is not None and batch_idx*BATCH_SIZE > sub_data_idxs[1]:
            break
        # shapes = np.array(shapes)
        images = images.to(device, dtype=torch.float)
        model.eval()
        with torch.no_grad():
            preds = model(images)
            preds = (nn.Sigmoid()(preds)>PREDICTION_THRESHOLD).double()
        preds = preds.cpu().numpy().astype(np.uint8)
        
        ls_masks = glob(data_dir + '/labels/*.tif')

        for i, pred in enumerate(preds):
            true_mask = preprocess_mask(ls_masks[batch_idx*BATCH_SIZE+i])
            shape = true_mask.shape
            pred = cv2.resize(pred[0], (shape[1], shape[0]), cv2.INTER_NEAREST)
            # pred = cv2.resize(pred[0], (shape[1], shape[0]), cv2.INTER_CUBIC)
             
            # pred = remove_small_objects(pred, 10)

            # The dimensions of the true mask and pred are [width, height] but need to be [1, width, height]
            # they need to stay as numpy arrays not torch tensors
            pred = np.expand_dims(pred, axis=0)
            true_mask = np.expand_dims(true_mask, axis=0)

            # print(f'pred shape: {pred.shape}, true_mask shape: {true_mask.shape}')

            # compute surface dice score
            surface_dice_score = fast_compute_surface_dice_score_from_tensor(pred, true_mask)
            surface_dice_scores.append(surface_dice_score)
    
    mean_surface_dice_score = np.mean(surface_dice_scores)
    return mean_surface_dice_score

def main():
    model = ImprovedUNet(in_channels=IN_CHANNELS, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)
    
    # save_predictions('data_downsampled512/train/kidney_1_voi', model, device=device, num=30)
    # save_predictions('data/train/kidney_1_voi', model, device=device, num=30)


    # ground_truth_rles = create_rle_df('saved_images/kidney_1_voi', subdir_name='labels')
    # prediction_rles = create_rle_df('saved_images/kidney_1_voi', subdir_name='preds')
    
    # # display(ground_truth_rles)
    # # print(ground_truth_rles.loc[0, 'height'])
    
    # surface_dice_score = score(ground_truth_rles, prediction_rles, 'id', 'rle')

    surface_dice_score = surface_dice(model, sub_data_idxs=(400, 500))
    # surface_dice_score = surface_dice(model)


    print(f'Surface Dice Score: {surface_dice_score:.4f}')
    
    # view_examples('saved_images/kidney_1_voi')
    
    # # calculate the dice score on the test set
    # dice_score = calc_dice_score(model, VAL_LOADER, device=device, verbose=True)
    # # dice_score = evaluate(model, VAL_LOADER, device=device, verbose=True, score_type='dice')
    # # dice_score = eval_and_plot(model, VAL_LOADER, device=device, verbose=True)
    # # dice_score, iou_score = evaluate(model, VAL_LOADER, device=device, verbose=True, score_type='both')
    # print(f'Validation Dice Score: {dice_score:.4f}')
    # # print(f'Validation IoU Score: {iou_score:.4f}')
    
    save_predictions_as_imgs(VAL_LOADER, model, num=30, folder='saved_images/', device=device)

    plot_samples(6, title='Predictions', include_image=True)

if __name__ == '__main__':
    main()