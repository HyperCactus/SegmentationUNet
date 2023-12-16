import os
import torch
import numpy as np
from glob import glob



# create a random tensor with shape (3, 4, 5)
# x = torch.rand(3, 4, 5)
# print(f'x: {x}')
# print(f'sum of x: {x.sum()}')

# mask = x > 0.5
# print(f'mask: {mask}')
# ones = torch.ones_like(x)
# print(f'masked ones: {ones[mask]}')
# print(f'number of ones: {ones[mask].sum()}')

# print(1 - (ones[mask].sum() / ones.sum()))

# targets = torch.rand(3, 4, 5)
# inputs = torch.rand(3, 4, 5)

# ones_in_mask = torch.sum(targets)
        
# # calculate the number of elements in the prediction that are greater than the threshold
# # ones_in_pred = (inputs > self.threshold).sum()
# mask = inputs > 0.6
# ones = torch.ones_like(inputs)
# ones_in_pred = torch.sum(ones[mask])

# loss = torch.abs(1 - (ones_in_pred / ones_in_mask))
# print(f'loss: {loss}')

DATASET_FOLDER = "data/kidney_1_dense"
ls_images = glob(os.path.join(DATASET_FOLDER, "test", "*", "*", "*.tif"))
print(f"found images: {len(ls_images)}")