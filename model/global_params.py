"""
Global parameters for the UNet model defined here for easy access across files.
"""

HPC = True
PNG_DSET = True
TEST_MODE = False#not HPC

# Hyperparameters
# LEARNING_RATE = 1e-4
LEARNING_RATE = 1e-5
BATCH_SIZE = 1 if HPC else 16
# BATCH_SIZE = 3 # Laptop
NUM_EPOCHS = 30 if HPC else 2
NUM_WORKERS = 1
PIN_MEMORY = True
PREDICTION_THRESHOLD = 0.53
IN_CHANNELS = 1 # I've stacked the previous image, current image, and next image as the input channels

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

HIGH_PASS_ALPHA = 0.1
HIGH_PASS_STRENGTH = 0.1

TILES_IN_X = 12#4
TILES_IN_Y = 9#3

TILE_SIZE = 256 if HPC else 32#64

CHECKPOINT_DIR = 'checkpoints/checkpoint.pth.tar'
# CHECKPOINT_DIR = 'checkpoints/in_chans3.pth.tar'
# CHECKPOINT_DIR = 'checkpoints/IoU_loss_15ep.pth.tar'
# CHECKPOINT_DIR = 'checkpoints/big_train.pth.tar'
# CHECKPOINT_DIR = 'checkpoints/1024_model.pth.tar'
# CHECKPOINT_DIR = 'checkpoints/256_model.pth.tar'
# CHECKPOINT_DIR = 'checkpoints/512_vol_train.pth.tar'

if PNG_DSET:
    BASE_PATH = 'data_png/train'
else:
    BASE_PATH = 'data_downsampled512/train' if HPC else 'data/train'

VAL_DATASET_DIR = BASE_PATH + '/kidney_3_dense'
VAL_IMG_DIR = VAL_DATASET_DIR + '/images'
VAL_MASK_DIR = VAL_DATASET_DIR + '/labels'

IMG_FILE_EXT = '.png' if PNG_DSET else '.tif'
MASK_FILE_EXT = '.png' if PNG_DSET else '.tif'

NOISE_MULTIPLIER = 0.0 if TEST_MODE else 0.01


# base_path = 'data_downsampled512/train'
# dataset = 'kidney_1_dense'
datasets = ['kidney_1_dense', 'kidney_1_voi', 'kidney_2', 'kidney_3_sparse']
TRAIN_DATASETS = ['kidney_1_dense']