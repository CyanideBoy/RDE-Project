import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import time

from model import ConvNet_Shallow_Single as ConvNet

import sys

import fasttext.CustomDataset as CustomDataset


tset = CustomDataset.get_dataset('data/test.csv')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = ConvNet(True, WINDOW=3, FEATURES=1)
print('Running on',device)
print('Building model..')
model.to(device)
print('Model Built.')

FILE = 'fasttext/sw5-1/Model_best_val_quicksave.pt'
BATCHSIZE = 2048

model.load_state_dict(torch.load(FILE))
model.eval()

test_points = len(tset)
print("Number of testing documents", test_points)
corr = 0.0


test_loader = torch.utils.data.DataLoader(tset, batch_size = BATCHSIZE,
                                            num_workers=8)

with torch.no_grad():
  for ib,sample in enumerate(test_loader):

    data_input = torch.as_tensor(sample['matrix'][:,None,:,:])
    data_output = torch.as_tensor(sample['class'][:,0])

    data_input = data_input.to(device, dtype=torch.float)
    data_output = data_output.to(device, dtype=torch.long)

    output = model(data_input)
    _, preds = torch.max(output,dim=1)
    corr += (torch.sum(preds == data_output.data)).data.item()

print('Test Accuracy: {:.2f}%'.format(corr*100.0/test_points))
#print(test_points)
#print(corr)
                                      
