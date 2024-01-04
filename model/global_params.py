"""
Global parameters for the UNet model defined here for easy access across files.
"""

HPC = True

# Hyperparameters
# LEARNING_RATE = 1e-4
LEARNING_RATE = 1e-8
BATCH_SIZE = 32 if HPC else 3
# BATCH_SIZE = 3 # Laptop
NUM_EPOCHS = 8 if HPC else 1
NUM_WORKERS = 4
PIN_MEMORY = True
PREDICTION_THRESHOLD = 0.6#0.5
IN_CHANNELS = 3 # I've stacked the previous image, current image, and next image as the input channels

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

HIGH_PASS_ALPHA = 0.1
HIGH_PASS_STRENGTH = 0.1

CHECKPOINT_DIR = 'checkpoints/checkpoint.pth.tar'

# BASE_PATH = 'data/train'
BASE_PATH = 'data_downsampled512/train' if HPC else 'data/train'

VAL_DATASET_DIR = BASE_PATH + '/kidney_2'
VAL_IMG_DIR = VAL_DATASET_DIR + '/images'
VAL_MASK_DIR = VAL_DATASET_DIR + '/labels'


# base_path = 'data_downsampled512/train'
# dataset = 'kidney_1_dense'
datasets = ['kidney_1_dense', 'kidney_1_voi', 'kidney_2', 'kidney_3_sparse']
TRAIN_DATASETS = ['kidney_1_dense', 'kidney_1_voi', 'kidney_3_sparse']