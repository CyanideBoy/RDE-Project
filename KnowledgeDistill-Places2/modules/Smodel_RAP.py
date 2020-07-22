from modules.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F



class StudentNet_2(nn.Module):

    def __init__(self, num_classes=365,sizes=None):
        super(StudentNet_2, self).__init__()
        
        if sizes is None:
            sizes= [32,64,128,128,32,    ##feature extractor
                        1024,512]         ##classifier
        
        self.features_1a = Sequential(
            Conv2d(3, sizes[0], kernel_size=12, stride=4, padding=2),   #55
            ReLU(inplace=True),
            AvgPool2d(kernel_size=3, stride=2))                         #27
            
        self.features_1b = Sequential(
            Conv2d(3, sizes[0], kernel_size=8, stride=2, padding=1),    #110
            ReLU(inplace=True),
            MaxPool2d(kernel_size=8, stride=4, padding=1))              #27
            
        self.features_2a = Sequential(
            Conv2d(2*sizes[0], sizes[1], kernel_size=5, padding=2),       #27
            ReLU(inplace=True),
            AvgPool2d(kernel_size=3, stride=2))                         #13
            
        self.features_2b = Sequential(
            Conv2d(2*sizes[0], sizes[1], kernel_size=7, padding=3),       #27
            ReLU(inplace=True),
            MaxPool2d(kernel_size=5, stride=2, padding=1))              #13
            
        self.features_3 = Sequential(
            Conv2d(2*sizes[1], sizes[2], kernel_size=3, padding=1),       #13
            ReLU(inplace=True),
            Conv2d(sizes[2], sizes[3], kernel_size=3, padding=1),       #13
            ReLU(inplace=True))
            
        self.features_4a = Sequential(
            Conv2d(sizes[3], sizes[4], kernel_size=3, padding=1),       #13
            ReLU(inplace=True),
            AvgPool2d(kernel_size=3, stride=2))                         #6
        
        self.features_4b = Sequential(
            Conv2d(sizes[3], sizes[4], kernel_size=5, padding=1),       #11
            ReLU(inplace=True),
            MaxPool2d(kernel_size=5, stride=2, padding=2))              #6
        

        self.avgpool = AdaptiveAvgPool2d((6, 6))

        self.classifier = Sequential(
            Dropout(0.3),
            Linear(2 * sizes[4] * 6 * 6, sizes[5]),
            ReLU(inplace=True),
            Dropout(0.2),
            Linear(sizes[5], sizes[6]),
            Dropout(0.2),
            ReLU(inplace=True),
            Linear(sizes[6], num_classes),
        )

    def forward(self, x):
        x1 = self.features_1a(x)
        x2 = self.features_1b(x)
        x = torch.cat((x1,x2),dim=1)
        x1 = self.features_2a(x)
        x2 = self.features_2b(x)
        x = torch.cat((x1,x2),dim=1)
        x1 = self.features_3(x)
        x = x + x1
        x1 = self.features_4a(x)
        x2 = self.features_4b(x)
        x = torch.cat((x1,x2),dim=1)
        
        #x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x    

    def RAP_relprop(self, R):
        R = self.fc.RAP_relprop(R)
        R = R.reshape_as(self.avgpool.Y)
        R = self.avgpool.RAP_relprop(R)

        R = self.layer4.RAP_relprop(R)
        R = self.layer3.RAP_relprop(R)
        R = self.layer2.RAP_relprop(R)
        R = self.layer1.RAP_relprop(R)

        R = self.maxpool.RAP_relprop(R)
        R = self.relu.RAP_relprop(R)
        R = self.bn1.RAP_relprop(R)
        R = self.conv1.RAP_relprop(R)

        return R