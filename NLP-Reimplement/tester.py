import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import time

from model import ConvNet
from CustomDataset import get_dataset

tset = get_dataset('20news-bydate-test')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = ConvNet(True)
print('Running on',device)
print('Building model..')	
model.to(device)
print('Model Built.')

FILE = 'weights/Model_quicksave30.pt'
BATCHSIZE = 256

model.load_state_dict(torch.load(FILE))
model.eval()

test_points = len(tset)
print("Number of testing documents", test_points)

corr = 0.0


test_loader = torch.utils.data.DataLoader(tset, batch_size = BATCHSIZE, 
                                            num_workers=8)

with torch.no_grad():
    for data_input, data_output, pafs in test_loader:
        
        data_input = torch.as_tensor(data_input, dtype=torch.float)
        data_output = torch.as_tensor(data_output)

        data_input = data_input.to(device)
        data_output = data_output.to(device)

        output = model(data_input)
        _, preds = torch.max(output,dim=1)
        corr += (torch.sum(preds == data_output.data)).data.item()
        
print('Test Accuracy: {:.2f}%'.format(corr*100.0/test_points))
print(test_points)
print(corr)