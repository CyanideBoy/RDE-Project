import numpy as np
import math

import matplotlib.pyplot as plt

import pickle
import argparse
import time
import itertools
from copy import deepcopy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils

batch_size = 256

os.environ['CUDA_VISIBLE_DEVICES'] = '3'    # set visible devices depending on system configuration
device = torch.device('cuda:0')

def reproducibilitySeed():
    """
    Ensure reproducibility of results; Seeds to 0
    """
    torch_init_seed = 0
    torch.manual_seed(torch_init_seed)
    numpy_init_seed = 0
    np.random.seed(numpy_init_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

reproducibilitySeed()


import torchvision
import torchvision.transforms as transforms

mnist_image_shape = (28, 28)
random_pad_size = 2

# Training images augmented by randomly shifting images by at max. 2 pixels in any of 4 directions
transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(mnist_image_shape, random_pad_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ]
            )

transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

train_val_dataset = torchvision.datasets.MNIST(root='./MNIST_dataset/', train=True, 
                                            download=True, transform=transform_train)

test_dataset = torchvision.datasets.MNIST(root='./MNIST_dataset/', train=False, 
                                            download=True, transform=transform_test)

num_train = int(1.0 * len(train_val_dataset) * 95 / 100)
num_val = len(train_val_dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [num_train, num_val])

train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


checkpoints_path = 'checkpoints_teacher/'
if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)
    
    
num_epochs = 100


learning_rates = list(np.logspace(-4, -2, 3))
#learning_rates = [1e-2]
learning_rate_decays = [0.95]    # learning rate decays at every epoch
#weight_decays = [0.0] + list(np.logspace(-5, -1, 5))
weight_decays = [1e-5]           # regularization weight
momentums = [0.9]
# dropout_probabilities = [(0.2, 0.5), (0.0, 0.0)]
dropout_probabilities = [(0.0, 0.0),(0.2,0.3),(0.3,0.2),(0.5,0.5)]

hparams_list = []
for hparam_tuple in itertools.product(dropout_probabilities, weight_decays, learning_rate_decays, 
                                        momentums, learning_rates):
    hparam = {}
    hparam['dropout_input'] = hparam_tuple[0][0]
    hparam['dropout_hidden'] = hparam_tuple[0][1]
    hparam['weight_decay'] = hparam_tuple[1]
    hparam['lr_decay'] = hparam_tuple[2]
    hparam['lr'] = hparam_tuple[4]
    hparams_list.append(hparam)
    

results = {}
for hparam in hparams_list:
    print('Training with hparams ' + utils.ToString(hparam))
    reproducibilitySeed()
    hparam_tuple = utils.DictToTuple(hparam)
    results[hparam_tuple] = utils.trainTeacher(hparam, num_epochs, 
                                                train_loader, val_loader,
                                                device=device)
    
    save_path = checkpoints_path + utils.ToString(hparam) + '_final.tar'
    torch.save({'results' : results[hparam_tuple][0], 
                'model_state_dict' : results[hparam_tuple][1], 
                'epoch' : num_epochs}, save_path)

# Calculate test accuracy
name = utils.TeacherIterate('checkpoints_teacher', test_loader, device, nn.CrossEntropyLoss())
np.save('best',name)
