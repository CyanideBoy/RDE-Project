import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import matplotlib.pyplot as plt
import time

from model import ConvNet
from CustomDataset import get_dataset
from util import plot_text_heatmap

import seaborn as sns
import pandas as pd

NUM_MAPS = 10
LEARNINGRATE = 8e-3
GAMMA = 0.9

BATCHSIZE = 512
NUMEPOCHS = 11

LOG_SOFTMAX_VALUES = False

dset = get_dataset('20news-bydate-test')

FILE = 'Model_quicksave39.pt'
#random.seed(12)
#np.random.seed(12)

class_names = dset.classes
print(class_names)


rand_files = random.sample(range(len(dset)), 4*NUM_MAPS)


model = ConvNet(LOG_SOFTMAX_VALUES)
model.load_state_dict(torch.load(FILE))
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model.to(device)
print('Model Built.')



print('Printing Parameters')
for name, param in model.named_parameters():
    param.requires_grad = False
    print(name)

print('---Printing Parameters Finished!---')


def GenRelMap(x, num_iters=400, lr =8e-3, gam = 0.95 , lamb = 750 ):


    s = 0.1*np.random.rand(1,x.shape[2],1)
    #s = 0.5*np.ones((1,x.shape[2],1))
    #s = np.zeros((1,x.shape[2],1))
    
    s = torch.as_tensor(s.astype(np.float32)).to(device)
    s = torch.autograd.Variable(s, requires_grad=True)

    
    optimizer = optim.Adam([s], lr = lr)             # OR RAdam/DiffGrad/etc
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gam)

    x_input = torch.as_tensor(x).to(device, dtype = torch.float)
    x_out = model(x_input.clone()).detach()
    highest_dim = int(np.argmax(x_out.cpu().numpy(), axis=1))
    #print(x_input.shape)
    print(highest_dim)
    for i in range(num_iters):

        n = torch.as_tensor(np.random.normal(size=(BATCHSIZE, *(x.shape[1:]))).astype(np.float32)).to(device)
        
        data_input = (x_input-n)*s+n
        out = model(data_input)

        loss = 0.5*torch.mean((out[:, highest_dim]-x_out[:, highest_dim])**2)+lamb*torch.mean(torch.abs(s))

        if (i)%50==0:
            print("Net loss:{:.5f}, MSE:{:.5f}, L1:{:.5f}".format(loss.data.item(),
                                                                  0.5*torch.mean((out[:, highest_dim]-x_out[:, highest_dim])**2).data.item(),
                                                                  lamb*torch.mean(torch.abs(s)).data.item()))
            print(list(np.argmax(out.detach().cpu().numpy(), axis=1)).count(highest_dim)*100.0/BATCHSIZE)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            s += s.clamp_(0,1) - s

            #print(s)
            #print(s.grad)
            #print(0.5*torch.mean((out[:, highest_dim]-x_out[:, highest_dim])**2).data.item())
            #print(lamb*torch.mean(torch.abs(s)).data.item())
          
        scheduler.step()

    return s.detach().cpu().numpy()




rmap = [None]*10
counter = 0
num = 0
while(counter < NUM_MAPS):

    e = dset.get_words_list(rand_files[num])

    sample, target, path = dset[rand_files[num]]
    sample = dset.get_custom_matrix(rand_files[num])
    sample = sample[None,None,:,:]


    print(path)
    print(len(e))
    print(target)

    x_input = torch.as_tensor(sample).to(device, dtype=torch.float)
    x_out = model(x_input.clone()).detach().cpu()

    num += 1

    if np.argmax(x_out) != target: 
        print("WRONG")
        print(x_out)
        print(np.argmax(x_out))
        continue

    rmap[counter] = (GenRelMap(sample, num_iters=501, lamb=7, gam=0.98, lr = 5e-3))[0,:,0]
    print(rmap[counter])

    #plot_text_heatmap(e,rmap[counter],counter)

    indx = sorted(range(len(rmap[counter])), key=lambda k: rmap[counter][k])
    wd = [e[i] for i in indx]
    wd = wd[::-1]
    wd = wd[:6]
    smap = [rmap[counter][i] for i in indx]
    smap = smap[::-1]
    smap = smap[:6]

    #pdf = pd.DataFrame(wd,columns=['Words'])
    #pdf['RMap'] = smap
    #sns.set(style="whitegrid")
    #ax = sns.barplot(x="Words", y="RMap", data=pdf)
    #plt.savefig('barplot_'+str(counter)+'.png')
    np.save(str(counter)+'.npy', [rmap[counter],e,wd,smap])
    counter += 1