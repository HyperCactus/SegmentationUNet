"""
Global parameters for the UNet model defined here for easy access across files.
"""

HPC = True
PNG_DSET = True
TEST_MODE = not HPC

# Hyperparameters
# LEARNING_RATE = 1e-4
LEARNING_RATE = 1e-4
BATCH_SIZE = 8 if HPC else 4
# BATCH_SIZE = 3 # Laptop
NUM_EPOCHS = 30 if HPC else 300
NUM_WORKERS = 4
PIN_MEMORY = True
PREDICTION_THRESHOLD = 0.6125#0.5
IN_CHANNELS = 1 # I've stacked the previous image, current image, and next image as the input channels

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

HIGH_PASS_ALPHA = 0.1
HIGH_PASS_STRENGTH = 0.1

TILES_IN_X = 4
TILES_IN_Y = 3

TILE_SIZE = 1024

# CHECKPOINT_DIR = 'checkpoints/checkpoint.pth.tar'
# CHECKPOINT_DIR = 'checkpoints/in_chans3.pth.tar'
# CHECKPOINT_DIR = 'checkpoints/IoU_loss_15ep.pth.tar'
CHECKPOINT_DIR = 'checkpoints/big_train.pth.tar'

if PNG_DSET:
    BASE_PATH = 'data_png/train'
else:
    BASE_PATH = 'data_downsampled512/train' if HPC else 'data/train'

VAL_DATASET_DIR = BASE_PATH + '/kidney_2'
VAL_IMG_DIR = VAL_DATASET_DIR + '/images'
VAL_MASK_DIR = VAL_DATASET_DIR + '/labels'

IMG_FILE_EXT = '.png' if PNG_DSET else '.tif'
MASK_FILE_EXT = '.png' if PNG_DSET else '.tif'

NOISE_MULTIPLIER = 0.0 if TEST_MODE else 0.01


# base_path = 'data_downsampled512/train'
# dataset = 'kidney_1_dense'
datasets = ['kidney_1_dense', 'kidney_1_voi', 'kidney_2', 'kidney_3_sparse']
TRAIN_DATASETS = ['kidney_1_dense', 'kidney_3_dense']