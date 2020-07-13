import numpy as np
import math

import matplotlib.pyplot as plt

import pickle
import argparse
import time
import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import model
import utils


batch_size = 512

os.environ['CUDA_VISIBLE_DEVICES'] = '2'    # set visible devices depending on system configuration
device = torch.device('cuda:0')


# Ensure reproducibility
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


mnist_image_shape = (28, 28)
random_pad_size = 2

# Training images augmented by randomly shifting images by at max. 2 pixels in any of 4 directions
transform_train = transforms.Compose(
                [   transforms.RandomCrop(mnist_image_shape, random_pad_size),
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

checkpoints_path_teacher = 'checkpoints_teacher/'
checkpoints_path_student = 'checkpoints_rap_change/'

if not os.path.exists(checkpoints_path_student):
    os.makedirs(checkpoints_path_student)
    
    
# set the hparams used for training teacher to load the teacher network
z = str(np.load('best.npy'))    
print('Loading ',z)
dropout_input = float(z.split(',')[1].split('=')[1])
dropout_hidden = float(z.split(',')[0].split('=')[1])
load_path = os.path.join(checkpoints_path_teacher,z)
teacher_net = model.TeacherNet(dropout_input,dropout_hidden)
teacher_net.load_state_dict(torch.load(load_path, map_location=device)['model_state_dict'])
teacher_net = teacher_net.to(device)

for param in teacher_net.parameters():
    param.requires_grad = False

# Calculate teacher test accuracy
_, test_accuracy = utils.getLossAccuracyOnDataset(teacher_net, test_loader, device)
print('teacher test accuracy: ', test_accuracy)

num_epochs = 100

#temperatures = [10]
#alphas = [(0.0,0.5)]  #(0.5,0.5),(0.8,0.2),(0.33,0.33),(0.0,1.0)
learning_rates = [1e-3]
learning_rate_decays = [0.95]
weight_decays = [0]
dropout_probabilities = [(0.0,0.0)]
hparams_list = []
#alphas, temperatures, 
for hparam_tuple in itertools.product(dropout_probabilities, weight_decays, learning_rate_decays, 
                                       learning_rates):
    hparam = {}
    #hparam['alpha_st'] = hparam_tuple[0][0]
    #hparam['alpha_rap'] = hparam_tuple[0][1]
    #hparam['T'] = hparam_tuple[1]
    hparam['dropout_input'] = hparam_tuple[0][0]
    hparam['dropout_hidden'] = hparam_tuple[0][1]
    hparam['weight_decay'] = hparam_tuple[1]
    hparam['lr_decay'] = hparam_tuple[2]
    hparam['lr'] = hparam_tuple[3]
    hparams_list.append(hparam)

results_distill_rap = {}
for hparam in hparams_list:
    print('Training with hparams' + utils.ToString(hparam))
    reproducibilitySeed()
    hparam_tuple = utils.DictToTuple(hparam)
    results_distill_rap[hparam_tuple] = utils.trainRAP2(teacher_net, hparam, num_epochs, 
                                                                train_loader, val_loader,  
                                                                device=device)
    save_path = checkpoints_path_student + utils.ToString(hparam) + '_final.tar'
    torch.save({'results' : results_distill_rap[hparam_tuple][0], 
                'model_state_dict' : results_distill_rap[hparam_tuple][1],
                'best_state_dict' : results_distill_rap[hparam_tuple][2], 
                'epoch' : num_epochs}, save_path)

utils.RAPIterate(checkpoints_path_student, test_loader, device, nn.CrossEntropyLoss())
