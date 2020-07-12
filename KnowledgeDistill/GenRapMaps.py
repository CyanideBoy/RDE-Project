import numpy as np
import torch
#import torchvision.transforms as transforms
#from torch.autograd import Variable as V
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from modules.resnet import resnet50
from modules.Smodel import StudentNet_2, StudentNet_1, AlexNet

from utils.tools import save_maps, compute_pred, fix
from utils.DataSet import ImageFolderWithPaths

import os
from PIL import Image
from tensorboardX import SummaryWriter
import math

import time
import sys
from pathlib import Path


print(torch.__version__)




def main():
    

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    cudnn.benchmark = True
    
    Batchsize_Train = 8
    Batchsize_Val = 8

    arch = 'resnet50'
    # load the pre-trained weights
    model_file = 'ptrained-models/%s_places365.pth.tar' % arch
    model = resnet50(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded RAP Enabled Model")
    model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False

    print('Model Built')

    # load the image transformer
    centre_crop = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = ImageFolderWithPaths('places365_standard/train', transform=centre_crop)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=Batchsize_Train,
                                                 shuffle=True, num_workers=4)

    val_data = ImageFolderWithPaths('places365_standard/val', transform=centre_crop)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=Batchsize_Val,
                                                 shuffle=True, num_workers=4)

    #print("Total training samples : {:,}".format(len(train_data)))
    #print("Total validation samples : {:,}".format(len(val_data)))
    
    classes = [x[0].split('/')[-1] for x in os.walk('places365_standard/train')]
    classes = classes[1:]
    #print(classes)
    '''
    file_name = 'categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    #print(classes)
    '''
    
    for i in range(365):
        if not os.path.exists('rap_maps/train/'+classes[i]):
            os.makedirs('rap_maps/train/'+classes[i])

        if not os.path.exists('rap_maps/val/'+classes[i]):
            os.makedirs('rap_maps/val/'+classes[i])

    for i, (inputs, labels, paths) in enumerate(train_loader):
        
        inputs = inputs.to(device)
        inputs.requires_grad = True
        labels = labels.to(device)

        output = model(inputs)
        T = compute_pred(output)
        T = T.to(device)
        #print(T.is_cuda)
        
        RAP = model.RAP_relprop(R=T)
        Res = (RAP).sum(dim=1, keepdim=True)
        
        heatmap = Res.permute(0, 2, 3, 1).data.cpu().numpy()
        paths = fix(paths)
        save_maps(heatmap.reshape([Batchsize_Train, 224, 224, 1]), paths)
        
        if i%(len(train_loader)//10) == 0:
            print('...working...')
        
    
    print('Train Set Done')
    for i, (inputs, labels, paths) in enumerate(val_loader):

        inputs = inputs.to(device)
        inputs.requires_grad = True
        labels = labels.to(device)

        output = model(inputs)
        T = compute_pred(output)
        T = T.to(device)
        #print(T.is_cuda)
        
        RAP = model.RAP_relprop(R=T)
        Res = (RAP).sum(dim=1, keepdim=True)
        
        heatmap = Res.permute(0, 2, 3, 1).data.cpu().numpy()
        paths = fix(paths)
        save_maps(heatmap.reshape([Batchsize_Val, 224, 224, 1]), paths)
        

if __name__ == '__main__':
    main()
