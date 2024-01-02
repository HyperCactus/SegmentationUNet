"""
Global parameters for the UNet model defined here for easy access across files.
"""
# Copied from COMP3710 report

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
# BATCH_SIZE = 3 # Laptop
NUM_EPOCHS = 5#30
NUM_WORKERS = 4
PIN_MEMORY = True
PREDICTION_THRESHOLD = 0.5
IN_CHANNELS = 1

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

HIGH_PASS_ALPHA = 0.1
HIGH_PASS_STRENGTH = 0.1

CHECKPOINT_DIR = 'checkpoints/checkpoint.pth.tar'

# BASE_PATH = 'data/train'
BASE_PATH = 'data_downsampled512/train'

VAL_DATASET_DIR = BASE_PATH + '/kidney_2'
VAL_IMG_DIR = VAL_DATASET_DIR + '/images'
VAL_MASK_DIR = VAL_DATASET_DIR + '/labels'


# base_path = 'data_downsampled512/train'
# dataset = 'kidney_1_dense'
datasets = ['kidney_1_dense', 'kidney_1_voi', 'kidney_2', 'kidney_3_sparse']
TRAIN_DATASETS = ['kidney_1_dense', 'kidney_1_voi', 'kidney_3_sparse']