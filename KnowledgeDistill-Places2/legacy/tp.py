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
    


    # load the image transformer
    centre_crop = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = ImageFolderWithPaths('places365_standard/train', transform=centre_crop)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8,
                                                 shuffle=True, num_workers=4)

    val_data = ImageFolderWithPaths('places365_standard/val', transform=centre_crop)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8,
                                                 shuffle=True, num_workers=4)

    
    print(len(train_loader))
    print(len(val_loader))
    
if __name__ == '__main__':
    main()
