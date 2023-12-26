
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