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


from modules.Smodel import StudentNet_2, StudentNet_1, AlexNet
from utils.tools import get_config, save_model,save_model_best, resume

import math

from PIL import Image

import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser
import sys
from pathlib import Path
import os
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/kd-config.yaml',
                    help="KD configuration")

def main():

    args = parser.parse_args()
    config = get_config(args.config)

    device_ids = config['gpu_ids']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = os.path.join('KD-Weights',config['method_name'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    

    Batchsize_Train = config['batch_size']
    Batchsize_Val = config['batch_size']


    teacher_arch = config['teacher']
    student_arch = 'StudentNet_2'  ## Useless

    # load the pre-trained weights
    Tmodel_file = 'ptrained-models/%s_places365.pth.tar' % teacher_arch
    Tmodel = models.__dict__[teacher_arch](num_classes=365)

    checkpoint = torch.load(Tmodel_file, map_location='cpu')
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

    Tmodel.load_state_dict(state_dict)
    Tmodel.eval()
    Tmodel.to(device)
    for param in Tmodel.parameters():
        param.requires_grad = False

    print('Teacher Model Built')

    Smodel = StudentNet_2()
    Smodel.to(device)
    print('Student Model Built')


    print('Total Parameters in Teacher Model : {:,}'.format(sum(p.numel() for p in Tmodel.parameters())))
    print('Total Parameters in Student_2 Model : {:,}'.format(sum(p.numel() for p in Smodel.parameters())))
    
    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.ImageFolder(config['train_data_path'], transform=centre_crop)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=Batchsize_Train,
                                                 shuffle=True, num_workers=4)

    val_data = datasets.ImageFolder(config['val_data_path'], transform=centre_crop)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=Batchsize_Val,
                                                 shuffle=True, num_workers=4)

    print("Total training samples : {:,}".format(len(train_data)))
    print("Total validation samples : {:,}".format(len(val_data)))


    LEARNINGRATE = config['lr']
    GAMMA = config['gamma']
    NUMEPOCHS = config['epochs']
    TEMPERATURE = config['temperature']
    LAMB = config['lambda']

    print('Initializing optimizer and scheduler..')

    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(Smodel.parameters(), lr = LEARNINGRATE)             # OR RAdam/DiffGrad/etc
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = GAMMA)

    print('Optimizer and scheduler initialized.')

    if config['pretrained']:
        print('Loading Pre-Trained Student Weights')
        ptrain_path = os.path.join('Weights',config['ptrain_name'])
        if not os.path.exists(ptrain_path):
            print('No Pre-Trained Model!')
            return
        opt = optim.Adam(Smodel.parameters(), lr = 1)  
        _1 ,Smodel, _2 = resume(ptrain_path, Smodel, opt, bv=True)

    writer = SummaryWriter(logdir=checkpoint_path)
    
    if config['resume']:
        start_epoch, Smodel, optimizer = resume(checkpoint_path,Smodel,optimizer,config['resume_bestval'])
    else:
        start_epoch = 1
    
    min_val_loss = float('+inf')
    for epoch in range(start_epoch, NUMEPOCHS+1):

        start_time = time.time()
        Smodel.train()

        tloss = 0.0
        tacc = 0

        total_train = 0
        flag_train = True

        for i, (inputs, labels) in enumerate(train_loader):

            if flag_train:
                if i > len(train_loader)//2 :
                    break

            inputs = inputs.to(device)
            labels = labels.to(device)

            Soup = Smodel(inputs)
            Soup_HT = F.log_softmax(Soup/TEMPERATURE,1)
            
            Toup = Tmodel(inputs)
            Toup_HT = F.softmax(Toup/TEMPERATURE, 1)
    
            loss = (1-LAMB)*ce_loss(Soup, labels) + LAMB*10*kl_loss(Soup_HT,Toup_HT)
            #print((1-LAMB)*ce_loss(Soup, labels))
            #print(LAMB*kl_loss(Soup_HT,Toup_HT))
            _, preds = torch.max(Soup, 1)
            tloss += loss.data.item() * inputs.data.size(0)
            tacc += (torch.sum(preds == labels.data)).data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += inputs.data.size(0)

        print('Trained on {} examples. Now, validating......'.format(total_train))

        vloss = 0
        vacc = 0

        Smodel.eval()
        with torch.no_grad():
            for i,(inputs,labels) in enumerate(val_loader):

                inputs = inputs.to(device)
                labels = labels.to(device)

                Soup = Smodel(inputs)
                #Soup_LT = F.softmax(Soup,1)
                Soup_HT = F.log_softmax(Soup/TEMPERATURE,1)
                
                Toup = Tmodel(inputs)
                Toup_HT = F.softmax(Toup/TEMPERATURE, 1)


                vloss += ((1-LAMB)*ce_loss(Soup, labels) + LAMB*kl_loss(Soup_HT,Toup_HT)).data.item()*inputs.size(0)
                _, preds = torch.max(Soup, 1)
                vacc += (torch.sum(preds == labels.data)).data.item()

        tloss /= total_train
        tacc = 100.0*tacc/total_train

        vloss /= len(val_data)
        vacc = 100.0*vacc/len(val_data)

        if vloss <= min_val_loss:
            min_val_loss = vloss
            save_model_best(Smodel,optimizer,checkpoint_path,epoch)

        stop_time = time.time()
        time_el = int(stop_time-start_time)
        print('epoch {}, train loss:{:.7f}, train acc {:.5f}, val loss:{:.7f}, val acc {:.5f} in {}h {}m {}s'.format(
                                                                                  epoch,
                                                                                  tloss, tacc,
                                                                                  vloss, vacc,
                                                                                  time_el//3600,
                                                                                  (time_el%3600)//60,
                                                                                  time_el%60))
        writer.add_scalars('Loss Curves', {'Train':tloss,
                                          'Val':vloss}, epoch)
        writer.add_scalars('Accuracy Curves', {'Train':tacc,
                                               'Val':vacc}, epoch)

        if epoch % config['snapshot_save_iter'] == 0:
                    save_model(Smodel,optimizer,checkpoint_path,epoch)

    writer.close()

if __name__ == '__main__':
    main()
