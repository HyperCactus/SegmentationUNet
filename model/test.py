import torch

print('Begin testing')
print('torch version: ', torch.__version__)
# testing cuda
if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda is NOT available')

x = torch.zeros(1).to('cuda')

print('Success')
