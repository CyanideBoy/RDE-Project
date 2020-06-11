import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch_optimizer as optim
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import time

from model import ConvNet


import sys

if len(sys.argv) == 2:
    if sys.argv[1] == 'word2vec':
        import word2vec.CustomDataset as CustomDataset
    elif sys.argv[1] == 'glove':
        import glove.CustomDataset as CustomDataset
    else:
        import fasttext.CustomDataset as CustomDataset
else:
    import fasttext.CustomDataset as CustomDataset



LEARNINGRATE = 3e-3
GAMMA = 0.98

BATCHSIZE = 32
NUMEPOCHS = 50


dset = CustomDataset.get_dataset('data/train.csv')
print('''
0 - World
1 - Sports
2 - Business
3 - Sci/Tech
''')

### TRAIN VAL SPLIT

NUM_DATA_POINTS = len(dset)

indices = list(range(NUM_DATA_POINTS))
split = int(np.floor(0.2 * NUM_DATA_POINTS))
np.random.shuffle(indices)

print(indices[0])


train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_points = len(train_indices)
val_points = len(val_indices)

train_loader = torch.utils.data.DataLoader(dset, batch_size = BATCHSIZE, 
                                           sampler=train_sampler, num_workers=4)
validation_loader = torch.utils.data.DataLoader(dset, batch_size = BATCHSIZE,
                                                sampler=valid_sampler, num_workers=4)

print("Printing Number of Documents Found")
print(NUM_DATA_POINTS)

## Loading Model

model = ConvNet(True, WINDOW = 3)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model.to(device)
print('Model Built.')

print('Initializing optimizer and scheduler..')

criterion = torch.nn.NLLLoss()
optimizer = optim.RAdam(model.parameters(), lr = LEARNINGRATE)             # OR RAdam/DiffGrad/etc
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = GAMMA)

print('Optimizer and scheduler initialized.')

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

model.apply(weights_init)

print('Printing Parameters')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
print('---Printing Parameters Finished!---')



min_val_loss = float('inf')

loss_values_train = []
loss_values_val = []

for epoch in range(1,NUMEPOCHS+1):
    start_time = time.time()
    
    model.train()
    runloss = 0.0
    tacc = 0
    
    for ib,sample in enumerate(train_loader):
        
        data_input = torch.as_tensor(sample['matrix'][:,None,:,:])
        data_output = torch.as_tensor(sample['class'][:,0])
        
        data_input = data_input.to(device, dtype=torch.float)  #, dtype=torch.float
        data_output = data_output.to(device, dtype=torch.long)    #, dtype=torch.long
        
        output = model(data_input) 
        _, preds = torch.max(output, 1)

        loss = criterion(output, data_output)
        runloss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        runloss += loss.item() * data_input.size(0)
        tacc += (torch.sum(preds == data_output.data)).data.item()

    runloss /= train_points 
    tacc = tacc*100.0/ train_points
    
    loss_values_train.append(runloss)
    
    model.eval()
    
    val_loss = 0
    corr = 0
    
    with torch.no_grad():
        for ib,sample in enumerate(validation_loader):
            
            data_input = torch.as_tensor(sample['matrix'][:,None,:,:])
            data_output = torch.as_tensor(sample['class'][:,0])

            data_input = data_input.to(device, dtype=torch.float)
            data_output = data_output.to(device, dtype=torch.long)

            output = model(data_input)
            _, preds = torch.max(output,dim=1)
            #print(output)
            #print(data_output)
            #print(_)
            #print(preds)
            #print(torch.sum(preds == data_output.data))
            
            loss = criterion(output, data_output)
            val_loss += loss.item()* data_input.size(0)
            corr += (torch.sum(preds == data_output.data)).data.item()
            

    val_loss /= val_points
    corr = corr*100.0/val_points
    loss_values_val.append(val_loss)

    if val_loss <= min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'fasttext/weights_3/Model_best_val_quicksave.pt')

    
    stop_time = time.time()
    time_el = int(stop_time-start_time)
    
    print('epoch [{}/{}], loss:{:.7f}, train acc:{:.4f}, val loss:{:.7f}, val acc:{:.4f} in {}h {}m {}s'.format(epoch, NUMEPOCHS,
                                                                                runloss, tacc, val_loss, corr,
                                                                                time_el//3600,
                                                                                (time_el%3600)//60,
                                                                               time_el%60))
    if epoch%5 ==0:
        torch.save(model.state_dict(), 'fasttext/weights_3/Model_quicksave'+str(epoch)+'.pt')

    scheduler.step()

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(np.array(loss_values_train), 'b')
plt.plot(np.array(loss_values_val), 'r')
plt.legend(['Train','Val'])
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('fasttext/train_curve_3.png')
