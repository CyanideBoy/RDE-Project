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

import model
import utils


batch_size = 512

os.environ['CUDA_VISIBLE_DEVICES'] = '3'    # set visible devices depending on system configuration
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


import torchvision
import torchvision.transforms as transforms

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
checkpoints_path_student = 'checkpoints_student2/'
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

# Calculate teacher test accuracy
_, test_accuracy = utils.getLossAccuracyOnDataset(teacher_net, test_loader, device)
print('teacher test accuracy: ', test_accuracy)

num_epochs = 100
'''
temperatures = [1]    # temperature for distillation loss
# trade-off between soft-target (st) cross-entropy and true-target (tt) cross-entropy;
# loss = alpha * st + (1 - alpha) * tt
alphas = [0.0]
learning_rates = [1e-3]
learning_rate_decays = [0.95]
weight_decays = [0]
dropout_probabilities = [(0.0, 0.0)]
hparams_list = []
for hparam_tuple in itertools.product(alphas, temperatures, dropout_probabilities, weight_decays, learning_rate_decays, 
                                        learning_rates):
    hparam = {}
    hparam['alpha'] = hparam_tuple[0]
    hparam['T'] = hparam_tuple[1]
    hparam['dropout_input'] = hparam_tuple[2][0]
    hparam['dropout_hidden'] = hparam_tuple[2][1]
    hparam['weight_decay'] = hparam_tuple[3]
    hparam['lr_decay'] = hparam_tuple[4]
    hparam['lr'] = hparam_tuple[5]
    hparams_list.append(hparam)
    
results_no_distill = {}
for hparam in hparams_list:
    print('Training with hparams' + utils.ToString(hparam))
    reproducibilitySeed()
    #student_net = model.StudentNetwork()
    #student_net = student_net.to(device)
    hparam_tuple = utils.DictToTuple(hparam)
    results_no_distill[hparam_tuple] = utils.trainStudent(teacher_net, hparam, num_epochs, 
                                                                    train_loader, val_loader, 
                                                                    device=device)
    save_path = checkpoints_path_student + utils.ToString(hparam) + '_final.tar'
    torch.save({'results' : results_no_distill[hparam_tuple][0], 
                'model_state_dict' : results_no_distill[hparam_tuple][1],
                'best_state_dict' : results_no_distill[hparam_tuple][2], 
                'epoch' : num_epochs}, save_path)
    
# Calculate student test accuracy
utils.StudentIterate('checkpoints_student2', test_loader, device, nn.CrossEntropyLoss())
'''
temperatures = [1,5,10,20]
# trade-off between soft-target (st) cross-entropy and true-target (tt) cross-entropy;
# loss = alpha * st + (1 - alpha) * tt
#alphas = [0.5,0.25,0.75]
alphas = [1.0]
learning_rates = [1e-3]
learning_rate_decays = [0.95]
weight_decays = [0]
dropout_probabilities = [(0.0,0.0)]
hparams_list = []
for hparam_tuple in itertools.product(alphas, temperatures, dropout_probabilities, weight_decays, learning_rate_decays, 
                                       learning_rates):
    hparam = {}
    hparam['alpha'] = hparam_tuple[0]
    hparam['T'] = hparam_tuple[1]
    hparam['dropout_input'] = hparam_tuple[2][0]
    hparam['dropout_hidden'] = hparam_tuple[2][1]
    hparam['weight_decay'] = hparam_tuple[3]
    hparam['lr_decay'] = hparam_tuple[4]
    hparam['lr'] = hparam_tuple[5]
    hparams_list.append(hparam)

results_distill = {}
for hparam in hparams_list:
    print('Training with hparams' + utils.ToString(hparam))
    reproducibilitySeed()
    #student_net = model.StudentNetwork()
    #student_net = student_net.to(fast_device)
    hparam_tuple = utils.DictToTuple(hparam)
    results_distill[hparam_tuple] = utils.trainStudent(teacher_net, hparam, num_epochs, 
                                                                train_loader, val_loader,  
                                                                device=device)
    save_path = checkpoints_path_student + utils.ToString(hparam) + '_final.tar'
    torch.save({'results' : results_distill[hparam_tuple][0], 
                'model_state_dict' : results_distill[hparam_tuple][1],
                'best_state_dict' : results_distill[hparam_tuple][2], 
                'epoch' : num_epochs}, save_path)

utils.StudentIterate('checkpoints_student2', test_loader, device, nn.CrossEntropyLoss())
