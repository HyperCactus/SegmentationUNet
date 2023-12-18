import torch

# testing cuda
if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda is NOT available')

import torch
torch.zeros(1).cuda()