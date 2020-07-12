import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print(torch.__version__)

from torchvision import transforms as trn
from torch.nn import functional as F
from torchvision import datasets, transforms

from tensorboardX import SummaryWriter

from modules.resnet import resnet50
from modules.Smodel_RAP import StudentNet_2

from utils.tools import get_config, save_model,save_model_best, resume
from utils.tools import save_maps, compute_pred, fix
from utils.DataSet import ImageFolderWithPaths, DatasetFolder

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
parser.add_argument('--config', type=str, default='configs/rap-config.yaml',
                    help="RAP configuration")

def main():

    args = parser.parse_args()
    config = get_config(args.config)

    device_ids = config['gpu_ids']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
    
    device_t = torch.device("cuda:0")
    device_s = torch.device("cuda:1")
    
    checkpoint_path = os.path.join('KD-Weights',config['method_name'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    

    Batchsize_Train = config['batch_size']
    Batchsize_Val = config['batch_size']


    teacher_arch = config['teacher']
    student_arch = 'StudentNet_2'  ## Useless

    # load the pre-trained weights
    Tmodel_file = 'ptrained-models/%s_places365.pth.tar' % teacher_arch
    Tmodel = resnet50(num_classes=365)
    
    
    checkpoint = torch.load(Tmodel_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    
    
    #print("Model's state_dict:")
    #for i, (a,b) in enumerate(zip(Tmodel.state_dict(),state_dict)):
    #    print(a,b)
        
    #for x in state_dict:
    #    print(x)
        
    Tmodel.load_state_dict(state_dict)
    Tmodel.eval()
    print("Loaded RAP Enabled Model")
    Tmodel.to(device_t)
    
    for param in Tmodel.parameters():
        param.requires_grad = False

    print('Teacher Model Built')

    Smodel = StudentNet_2()
    Smodel.to(device_s)
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
    LAMB_KL = config['lambda_kl']
    LAMB_RAP = config['lambda_rap']

    print('Initializing optimizer and scheduler..')

    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss()
    mse_loss = torch.nn.MSELoss(reduction='sum')
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

            inputs_t = inputs.to(device_t)
            inputs_s = inputs.to(device_s) 
            inputs_t.requires_grad = True
            inputs_s.requires_grad = True
            labels_t = labels.to(device_t)
            labels_s = labels.to(device_s)
            
            Toup = Tmodel(inputs_t)
            T = compute_pred(Toup).to(device_t)
            RAP = Tmodel.RAP_relprop(R=T)
            Res = (RAP).sum(dim=1, keepdim=True)
            heatmap_t = Res.permute(0, 2, 3, 1).data.cpu().numpy()
            print(heatmap_t.shape)
            heatmap_t.reshape([Batchsize_Train, -1])
            
            _t, preds_t = torch.max(Toup, 1)
            preds_t == labels_t
            preds_t = preds_t.cpu().numpy()
            Toup_HT = F.softmax(Toup/TEMPERATURE, 1)
            
            #Soup_LT = F.softmax(Soup, 1)
            Soup = Smodel(inputs_s)
            T = compute_pred(Soup).to(device_s)
            RAP = Smodel.RAP_relprop(R=T)
            Res = (RAP).sum(dim=1, keepdim=True)
            heatmap_s = Res.permute(0, 2, 3, 1).data.cpu().numpy()
            print(heatmap_s.shape)
            heatmap_s.reshape([Batchsize_Train, -1])
            
            Soup_HT = F.softmax(Soup/TEMPERATURE,1)
            
            loss_ce = ce_loss(Soup, labels_s)
            loss_mse =  LAMB_KL*kl_loss(Soup_HT,Toup_HT.to(device_s))
            loss_rap = LAMB_RAP*mse_loss(torch.Tensor(preds_t*heatmap_t).to(device_t), torch.Tensor(preds_t*heatmap_s).to(device_t))/np.sum(preds_t)
            
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
                Soup_HT = F.softmax(Soup/TEMPERATURE,1)
                
                Toup = Tmodel(inputs)
                Toup_HT = F.softmax(Toup/TEMPERATURE, 1)


                vloss += (ce_loss(Soup, labels) + LAMB*mse_loss(Soup_HT,Toup_HT)).data.item()*inputs.size(0)
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
