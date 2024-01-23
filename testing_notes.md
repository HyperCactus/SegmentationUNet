
* 2D improved UNet (256x256), Loss = (0.5*BCE_logits + 0.5*focal_loss + 0.5*IoU_loss), 4 Epochs
validation dice score = 0.1408

* Same but with just IoU loss
validation dice score = 0.2034

* 8 epochs with Loss = (0.1*BCE_logits + 0.2*focal_loss + 0.7*IoU_loss)
validation dice score = 0.3440, consistent degrease in loss

* idea: change random crop to and resize so that it can crop up to the maximum given the resolution and resize resolution
for example if the resize resolution is 256x256, the maximum zoom of the random crop is until the resolution of the crop
is 256x256

* idea: add anti-aliasing in encoder section of model, that way the model can handle higher resolution input.
* 8 epochs with Loss = (0.3*BCE_logits + 0.7*IoU_loss) now at 512x512
validation dice score = 0.5871 (seems like resolution is too low)

* put crop before resize, everything else same as last time.
validation dice score = 0.6405

* Added attention gates as per https://arxiv.org/pdf/1804.03999.pdf and trained for 8 epochs, (batch size 4 instead of due to increased memory consumption)
validation dice score = 0.6786 ~0.6915

* ImporvedAttentionUNet 14 epochs, 512x512 resolution
validation dice score = 0.7154

* idea: apply high pass filter and/or contrast increase and/or some other pre-processing to make it easier for the network to learn important features.
  
* added emboss augmentation (high pass filter)
validation dice score = 0.5186 (The scoring metric is acting sus, sometimes it gives significantly different scores for same model, idk why)

* ToDo: implement TTA, averaging predictions over multiple augmented versions of an image. Need to make now dataset class for this
* ToDo: investigate edge loss and Hausdorff distance loss
* ToDo: make 3D version of model
* ToDo: compute submission and surface dice score
* ToDo: train on rangpur
* Note: small vanes are not segmented, need to force the model to segment small vanes with some loss that is not just proportional to number of correctly segmented pixels
  
* trained on full dataset, 8 epochs
validation dice score = 0.6145 (unreliable)

* idea: pass multiple sequential images at a time to give 3D context, all with attention gates 
* trained more, 15 epochs total
validation dice score = 0.6541

* dataset compressed, no mask files, need to use RLE masks
* Need to redo dataset, this one does something stupid with channels
* ToDo: Integrate TensorBoard for testing on rangpur
* To remove extra channels in image import use ` image = cv2.imread('image.tif', cv2.CV_LOAD_IMAGE_GRAYSCALE)` ps " looks like cv2.IMREAD_GRAYSCALE is the way to go these days instead of cv2.CV_LOAD_IMAGE_GRAYSCALE" just look [here](https://stackoverflow.com/questions/18870603/in-opencv-python-why-am-i-getting-3-channel-images-from-a-grayscale-image)
* fix cuda problem on rangpur? ...  CUDA 10.2
`conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch`
* Downgrading python version on Rangpur, original version = Python 3.11.4, new version = Python 3.7.1
* Start recording train time as well, could try loading entire downsampled dataset into memory to speed up training
* Now running successfully on Rangpur!, ToDo: hpyerparamiter search, 3D upgrade, TTA, surface dice evaluation, better loss function
* Trained on Rangpur for 30 epochs with downsampled dataset
validation dice score = 0.7964
surface dice score = 0.9564
* Check if overfitting!
* implement smart tiling! check out https://www.kaggle.com/code/squidinator/sennet-hoa-in-memory-tiled-dataset-pytorch
* need to resize masks to original size for submission, tiling is the ultimate goal
* check [this](https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch) out for gpu memory usage info for rangpur optimization.
* submission to kaggle, images downsampled to 512x512 and then upscaled using cv2.INTER_CUBIC interpolation, using remove small objects 5.
public score: 0.208
* turned off remove small objects.
public score: 0.209
* Local inference using kidney 2 as validation set with same process as kaggle submission, prediction threshold 0.9.
Surface Dice Score: 0.6954
* Average train time on Rangpur = 3 min per epoch
* Added shear augmentations 

* for tta: `pred = model(non augmented)`
`pred *= model(tta image)/THRESHOLD`

* added simple tiling:
public score -> 0.474

* added advanced tiling with averaging on overlap, increasing tile size is like TTA, tiles = 6x6 (on subset of k2):
local Surface Dice Score: 0.9749

* advanced tiling, no TTA, 6x6 tiles:
public score -> 0.524

* local 3D Surface Dice Score computation on k2 subset with TTA and 5x4 tiles, using min max norm:
3D Surface Dice Score: 0.8113647134695395

* local 3D Surface Dice Score computation on k2 subset with TTA and 5x4 tiles, using /255 norm:
3D Surface Dice Score: 0.8560464274635169

* local 3D Surface Dice Score computation on k2 subset with TTA and 8x6 tiles, using /255 norm:
3D Surface Dice Score: 0.8622700067466642

* local 3D Surface Dice Score computation on k2 subset with TTA and 3x3 tiles, using /255 norm:
3D Surface Dice Score: 0.8295686825426427

* local 3D Surface Dice Score computation on k2 subset with TTA and 3x3 tiles, using /255 norm with remove small objects 5:
3D Surface Dice Score: 0.8319619583782941
* local 3D Surface Dice Score computation on k2 subset with TTA and 3x3 tiles, using /255 norm with remove small objects 10:
3D Surface Dice Score: 0.8179505718489243
* local 3D Surface Dice Score computation on k2 subset with TTA and 3x3 tiles, using /255 norm with remove small objects 3:
3D Surface Dice Score: 0.831656864880501

* Normalization, chop outliers, make kidney invariant

* Rangpur Partitions:
PARTITION
cpu
largecpu
p100
kaleen
a100
a100-test*

* test with/without noise, 3 vs 1 in_chans, ..., png data on rangpur 
* try multiple optimizers with different loss functions...
* add multiscale inference pipeline

* experiment result, iou loss, 15 epoch, on val set (500, 1400)
3D SDC = 0.7060
* binary dice loss, 15 epoch, on val set (500, 1400)
3D SDC = 0.6105