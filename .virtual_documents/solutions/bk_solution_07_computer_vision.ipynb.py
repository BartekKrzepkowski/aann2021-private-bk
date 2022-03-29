get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")


from itertools import product

import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.utils import make_grid


dataset = CIFAR10(root='.', train=True, transform=ToTensor(), download=True)
loader = DataLoader(dataset, batch_size=15, shuffle=True)

# get a single batch
x, y = next(iter(loader))
    
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(30, 15))
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        ind = 5 * i + j
        ax.imshow(np.transpose(x[ind], (1, 2, 0)))
        ax.set_title(f"{dataset.classes[y[ind]]}", fontdict={'size': 30})
        ax.set_xticks([])
        ax.set_yticks([])

fig.tight_layout()

print(f"CIFAR-10 classes: {dataset.classes}")


def loaders(batch_size, exp_no=0):
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CIFAR10(root='.', 
                            train=True,
                            download=True,
                            transform=transforms)

    test_dataset = CIFAR10(root='.', 
                           train=False,
                           download=True,
                           transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    loaders = {
        'train': train_loader,
        'test': test_loader
    }
    return loaders


from trainer import Dropout, BatchNorm1d, BatchNorm2d

class BN2d_Conv_Net(nn.Module):
    def __init__(self):
        super(BN2d_Conv_Net, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 15
                                 BatchNorm2d(32),
                                 nn.ReLU(),
                                 Dropout(0.2),
                                 nn.Conv2d(32, 64, kernel_size=3, stride=1), # 13
                                 BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 6
                                 BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2, stride=2)) #3
        self.fc = nn.Sequential(Dropout(0.2),
                                nn.Linear(3*3*128, 256),
                                BatchNorm1d(256),
                                nn.ReLU(),
                                Dropout(0.3),
                                nn.Linear(256, 100),
                                BatchNorm1d(100),
                                nn.ReLU(),
                                nn.Linear(100, 10))

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


from trainer import Trainer, IteratorParams

params_trainer = {
    'model': BN2d_Conv_Net,
    'loaders': loaders,
    'criterion': torch.nn.CrossEntropyLoss,
    'optim': torch.optim.Adam
}

trainer = Trainer(**params_trainer)


wd_arr = np.concatenate([np.cumproduct([0.1]*3), np.cumproduct([0.1]*3)/2]) / 10
lr_arr = np.concatenate([np.cumproduct([0.1]*3), np.cumproduct([0.1]*3)/2]) / 10

model_ls = [{}]
loaders_ls = [{'exp_no':0, 'batch_size':100}]
criterion_ls = [{}]
optim_ls = [{'lr': round(lr, 6), 'weight_decay': round(wd, 6), 'amsgrad': True} for lr in lr_arr for wd in wd_arr]

iter_params = IteratorParams(model_ls, loaders_ls, criterion_ls, optim_ls)

params_runs = {
    'iter_params': iter_params,
    'epochs': 5,
    'exp_name': 'init',
    'key_params': ['lr', 'weight_decay'],
    'device': device
}

trainer.run_trainer(**params_runs)


model_ls = [{}]
loaders_ls = [{'exp_no':0, 'batch_size':100}]
criterion_ls = [{}]
optim_ls = [{'lr': 0.001, 'weight_decay': 0.0005, 'amsgrad': True}]

iter_params = IteratorParams(model_ls, loaders_ls, criterion_ls, optim_ls)

params_runs = {
    'iter_params': iter_params,
    'epochs': 30,
    'exp_name': 'proper_training',
    'key_params': None,
    'device': device
}

trainer.run_trainer(**params_runs)


import pandas as pd

pd.read_csv('data/proper_training_2021-12-10_23-39-48/proper_training.csv')


from torchvision.transforms import RandomVerticalFlip

dataset = CIFAR10(root='.', train=True, transform=ToTensor())
loader = DataLoader(dataset, batch_size=15, shuffle=True)

flip = RandomVerticalFlip(p=1) 

# get a single batch
x, y = next(iter(loader))

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(30, 15))

for img, label, ax in zip(x, y, axes[0]):
    ax.imshow(np.transpose(img, (1, 2, 0)))
    ax.set_title(f"{dataset.classes[label]}", fontdict={'size': 30})
    ax.set_xticks([])
    ax.set_yticks([])
    
for img, label, ax in zip(x, y, axes[1]):
    ax.imshow(np.transpose(flip(img), (1, 2, 0)))
    ax.set_title(f"Flipped {dataset.classes[label]}", fontdict={'size': 30})
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()


# load in the data with the augumentations here
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ColorJitter

def loaders_aug(batch_size, exp_no=0):
    transforms = Compose([ToTensor(),
                          RandomHorizontalFlip(),
                          ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                          Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CIFAR10(root='.', 
                            train=True,
                            download=True,
                            transform=transforms)

    test_dataset = CIFAR10(root='.', 
                           train=False,
                           download=True,
                           transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    loaders = {
        'train': train_loader,
        'test': test_loader
    }
    return loaders


params_trainer = {
    'model': BN2d_Conv_Net,
    'loaders': loaders_aug,
    'criterion': torch.nn.CrossEntropyLoss,
    'optim': torch.optim.Adam
}

trainer = Trainer(**params_trainer)

model_ls = [{}]
loaders_ls = [{'exp_no':0, 'batch_size':100}]
criterion_ls = [{}]
optim_ls = [{'lr': 0.001, 'weight_decay': 0.0005, 'amsgrad': True}]

iter_params = IteratorParams(model_ls, loaders_ls, criterion_ls, optim_ls)

params_runs = {
    'iter_params': iter_params,
    'epochs': 30,
    'exp_name': 'augmentation',
    'key_params': None,
    'device': device
}

trainer.run_trainer(**params_runs)


pd.read_csv('data/augmentation_2021-12-10_23-46-54/augmentation.csv')


cifar_sample = np.load('resources/cifar_sample.npy')
np_image = cifar_sample[0]
image = torch.tensor(np_image)
plt.axis('off')
_ = plt.imshow(np_image.transpose(1,2,0)) 


def max_pooling(image: torch.tensor, 
                kernel_size: int, 
                stride: int = 1, 
                padding: int = 1):
    """
    :param image: torch.Tensor 
        Input image of shape (C, H, W)
    :param kernel_size: int 
        Size of the square pooling kernel
    :param stride: int
        Stride to use in pooling
    :param padding: int
        Zero-padding to add on all sides of the image 
    """
    # get image dimensions
    img_channels, img_height, img_width = image.shape
    # calculate the dimensions of the output image
    out_height = (img_height - kernel_size + 2 * padding) // stride + 1
    out_width = (img_height - kernel_size + 2 * padding) // stride + 1
    out_channels = img_channels
    
    if padding > 0:
        image_padded = torch.zeros((img_channels, img_height+2*padding, img_width+2*padding))
        image_padded[:, padding: -padding, padding: -padding] = image
        image = image_padded
    
    receptive_field = torch.tensor([image[c, i*stride: i*stride + kernel_size, j*stride: j*stride + kernel_size].flatten().numpy()
                           for c in range(img_channels) for i in range(out_height) for j in range(out_width)]).T
                
    max_pooled = torch.max(receptive_field, axis=0)[0].reshape(out_channels, out_height, out_width)

    return max_pooled


kernel_sizes = [4, 5, 6]
paddings = [0, 1, 2]
strides = [1, 2, 3, 4]

for kernel_size, stride, padding in product(kernel_sizes, strides, paddings):
    out = max_pooling(image, kernel_size=kernel_size, stride=stride, padding=padding)
    out_torch = torch.nn.functional.max_pool2d(input=image.unsqueeze(0), kernel_size=kernel_size, padding=padding, stride=stride)
    assert out_torch.squeeze().shape == out.shape
    assert torch.allclose(out, out_torch.squeeze(), atol=1e-5, rtol=1e-5)


def convolution(image: torch.tensor, 
                filters: torch.tensor, 
                biases: torch.tensor, 
                stride: int = 1, 
                padding: int = 1):
    """
    :param image: torch.Tensor 
        Input image of shape (C, H, W)
    :param filters: torch.Tensor 
        Filters to use in convolution of shape (K, C, F, F)
    :param biases: torch.Tensor 
        Bias vector of shape (K,)
    :param stride: int
        Stride to use in convolution
    :param padding: int
       Zero-padding to add on all sides of the image 
    """
    # get image dimensions
    img_channels, img_height, img_width = image.shape 
    n_filters, filter_channels, filter_size, filter_size = filters.shape 
    # calculate the dimensions of the output image
    out_height = (img_height - filter_size + 2 * padding) // stride + 1
    out_width = (img_height - filter_size + 2 * padding) // stride + 1
    out_channels = n_filters
    
    if padding > 0:
        image_padded = torch.zeros((img_channels, img_height+2*padding, img_width+2*padding))
        image_padded[:, padding: -padding, padding: -padding] = image
        image = image_padded

    receptive_field = torch.tensor([image[:, i*stride: i*stride + filter_size, j*stride: j*stride + filter_size].flatten().numpy() # [out_height * out_height, C * F * F]
                                    for i in range(out_height) for j in range(out_width)]).T # [C * F * F, out_height * out_width]
    
    filters = filters.flatten(start_dim=1) # [K, C * F * F]
    conv2d = filters @ receptive_field + biases.unsqueeze(-1)
    conv2d = (conv2d).reshape(out_channels, out_height, out_width)
    return conv2d


paddings = [0, 1, 2, 3]
strides = [1, 2, 3, 4]
filters = [(torch.randn((2,3,3,3)), torch.ones((2))),
           (torch.randn((2,3,5,5)), torch.ones((2))),
           (torch.randn((5,3,1,1)), torch.ones((5)))]

for (filt, bias), stride, padding in product(filters, strides, paddings):
    out = convolution(image, filt, bias, stride=stride, padding=padding)
    out_torch = torch.conv2d(input=image.unsqueeze(0), weight=filt, bias=bias, padding=padding, stride=stride)
    assert out_torch.squeeze().shape == out.shape
    assert torch.allclose(out, out_torch.squeeze(), atol=1e-5, rtol=1e-5)



