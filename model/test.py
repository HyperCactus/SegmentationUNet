import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from modules import ImprovedUNet
from glob import glob
from utils import inference_fn, load_checkpoint, plot_examples
from dataset import preprocess_image, preprocess_mask
from global_params import *
        
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedUNet(in_channels=IN_CHANNELS, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)

    plot_examples(model, num=5, device='cpu', dataset_folder=VAL_DATASET_DIR, sub_data_idxs=(500, 1400))
