"""
Global parameters for the UNet model defined here for easy access across files.
"""
# Copied from COMP3710 report

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_WORKERS = 4
PIN_MEMORY = True
PREDICTION_THRESHOLD = 0.5

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

HIGH_PASS_ALPHA = 0.1
HIGH_PASS_STRENGTH = 0.1

CHECKPOINT_DIR = 'checkpoints/checkpoint.pth.tar'

base_path = 'data/train'
# base_path = 'data_downsampled512/train'
# dataset = 'kidney_1_dense'
datasets = ['kidney_1_dense', 'kidney_1_voi', 'kidney_2', 'kidney_3_sparse']