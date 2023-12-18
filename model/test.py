import torch

print('Begin testing')
print('torch version: ', torch.__version__)
# testing cuda
if torch.cuda.is_available():
    print('cuda is available')
else:
    print('cuda is NOT available')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.zeros(1)
print(f'x on cpu: {x}')
x = x.to(device)
print(f'x on {device}: {x}')

print('Success')
