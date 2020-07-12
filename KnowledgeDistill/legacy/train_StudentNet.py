import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print(torch.__version__)

import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from torchvision import datasets, transforms

from tensorboardX import SummaryWriter


from Smodel import StudentNet_2, StudentNet_1, AlexNet

from PIL import Image

import matplotlib.pyplot as plt
import time

import sys
from pathlib import Path



teacher_arch = 'resnet50'
student_arch = 'StudentNet'  ## Useless


## GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Batchsize_Train = 1024
Batchsize_Val = 1024



# load the pre-trained weights
Tmodel_file = '%s_places365.pth.tar' % teacher_arch

Tmodel = models.__dict__[teacher_arch](num_classes=365)
#Tmodel.to(device)


checkpoint = torch.load(Tmodel_file, map_location='cpu')#os.system('wget ' + weight_url)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

Tmodel.load_state_dict(state_dict)
Tmodel.eval()

print('Teacher Model Built')

Smodel = StudentNet_2()
Smodel.to(device)
print('Student Model Built')


print('Total Parameters in Teacher Model : {:,}'.format(sum(p.numel() for p in Tmodel.parameters())))
print('Total Parameters in Student_2 Model : {:,}'.format(sum(p.numel() for p in Smodel.parameters())))
print('Total Parameters in Student_1 Model : {:,}'.format(sum(p.numel() for p in StudentNet_1().parameters())))
print('Total Parameters in AlexNet Model : {:,}'.format(sum(p.numel() for p in AlexNet().parameters())))

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

file_name = 'categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

train_data = datasets.ImageFolder('places365_standard/train', transform=centre_crop)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=Batchsize_Train,
                                             shuffle=True, num_workers=8)


val_data = datasets.ImageFolder('places365_standard/val', transform=centre_crop)

val_loader = torch.utils.data.DataLoader(val_data, batch_size=Batchsize_Val,
                                             shuffle=True, num_workers=8)

print("Total training samples : {:,}".format(len(train_data)))
print("Total validation samples : {:,}".format(len(val_data)))


LEARNINGRATE = 1e-3
GAMMA = 0.91
NUMEPOCHS = 20


print('Initializing optimizer and scheduler..')

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(Smodel.parameters(), lr = LEARNINGRATE)             # OR RAdam/DiffGrad/etc
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = GAMMA)

print('Optimizer and scheduler initialized.')


writer = SummaryWriter(logdir='logs/SN2-vanilla-smallstep')
Path("/home/SharedData3/ushasi/tub/gan/Weights/SN2-vanilla-smallstep").mkdir(parents=True, exist_ok=True)

min_val_loss = float('+inf')
for epoch in range(1,NUMEPOCHS+1):
    
    start_time = time.time()
    Smodel.train()
    
    tloss = 0.0
    tacc = 0
    
    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        output = Smodel(inputs)

        loss = criterion(output, labels)
        _, preds = torch.max(output, 1)
        tloss += loss.data.item() * inputs.data.size(0)
        tacc += (torch.sum(preds == labels.data)).data.item()
        
        #print(inputs.size(0))
        #print((torch.sum(preds == labels.data)).data.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    vloss = 0
    vacc = 0
    
    Smodel.eval()
    with torch.no_grad():
        for i,(inputs,labels) in enumerate(val_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            output = Smodel(inputs)
            
            
            vloss += criterion(output, labels).data.item()*inputs.size(0)
            _, preds = torch.max(output, 1)
            vacc += (torch.sum(preds == labels.data)).data.item()
            
            #print(inputs.size(0)) 
            #print(preds.size)
            #print(labels.size)
            #print((torch.sum(preds == labels.data)).data.item())
        
    # saving model if validation loss is lowest
    tloss /= len(train_data)
    tacc = 100.0*tacc/len(train_data)
    
    vloss /= len(val_data)
    vacc = 100.0*vacc/len(val_data)
    
    if vloss <= min_val_loss:
        min_val_loss = vloss
        torch.save(Smodel.state_dict(), 'Weights/SN2-vanilla-smallstep/Model_best_val.pt')

    # printing some stuff
    stop_time = time.time()
    time_el = int(stop_time-start_time)
    print('epoch {}, train loss:{:.7f}, train acc {:.5f}, val loss:{:.7f}, val acc {:.5f} in {}h {}m {}s'.format(
                                                                              epoch,
                                                                              tloss, tacc,
                                                                              vloss, vacc,
                                                                              time_el//3600,
                                                                              (time_el%3600)//60,
                                                                              time_el%60))
    writer.add_scalar('Train Loss',tloss,epoch)
    writer.add_scalar('Val Loss',vloss,epoch)
    writer.add_scalar('Train Acc',tacc,epoch)
    writer.add_scalar('Val Acc',vacc,epoch)
    
    torch.save(Smodel.state_dict(), 'Weights/SN2-vanilla-smallstep/Model_epoch_'+str(epoch)+'.pt')
    
writer.close()
