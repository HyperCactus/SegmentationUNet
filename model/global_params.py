"""
Global parameters for the UNet model defined here for easy access across files.
"""
# Copied from COMP3710 report

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 6
NUM_EPOCHS = 6
NUM_WORKERS = 4
PIN_MEMORY = True

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# Downsampled data set paths
TRAIN_IMG_DIR = 'data/ISIC_2017_downsampled/train/images'
TRAIN_MASK_DIR = 'data/ISIC_2017_downsampled/train/masks'

TEST_IMG_DIR = 'data/ISIC_2017_downsampled/test/images'
TEST_MASK_DIR = 'data/ISIC_2017_downsampled/test/masks'

VAL_IMG_DIR = 'data/ISIC_2017_downsampled/val/images'
VAL_MASK_DIR = 'data/ISIC_2017_downsampled/val/masks'

base_path = 'data/train/'  
dataset = 'kidney_1_dense/'