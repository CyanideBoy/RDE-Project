import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class StudentNet_2(nn.Module):

    def __init__(self, num_classes=365,sizes=None):
        super(StudentNet_2, self).__init__()
        
        if sizes is None:
            sizes= [32,64,128,128,32,    ##feature extractor
                        1024,512]         ##classifier
        
        self.features_1a = nn.Sequential(
            nn.Conv2d(3, sizes[0], kernel_size=12, stride=4, padding=2),   #55
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2))                         #27
            
        self.features_1b = nn.Sequential(
            nn.Conv2d(3, sizes[0], kernel_size=8, stride=2, padding=1),    #110
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=4, padding=1))              #27
            
        self.features_2a = nn.Sequential(
            nn.Conv2d(2*sizes[0], sizes[1], kernel_size=5, padding=2),       #27
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2))                         #13
            
        self.features_2b = nn.Sequential(
            nn.Conv2d(2*sizes[0], sizes[1], kernel_size=7, padding=3),       #27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=1))              #13
            
        self.features_3 = nn.Sequential(
            nn.Conv2d(2*sizes[1], sizes[2], kernel_size=3, padding=1),       #13
            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[2], sizes[3], kernel_size=3, padding=1),       #13
            nn.ReLU(inplace=True))
            
        self.features_4a = nn.Sequential(
            nn.Conv2d(sizes[3], sizes[4], kernel_size=3, padding=1),       #13
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2))                         #6
        
        self.features_4b = nn.Sequential(
            nn.Conv2d(sizes[3], sizes[4], kernel_size=5, padding=1),       #11
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2))              #6
        

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2 * sizes[4] * 6 * 6, sizes[5]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(sizes[5], sizes[6]),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(sizes[6], num_classes),
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
    
    
class AlexNet(nn.Module):

    def __init__(self, num_classes=365):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
    
class StudentNet_1(nn.Module):

    def __init__(self, num_classes=365,sizes=None):
        super(StudentNet_1, self).__init__()
        
        if sizes is None:
            sizes= [64,128,128,128,64,    ##feature extractor
                        1024,512]         ##classifier
        
        self.features = nn.Sequential(
            nn.Conv2d(3, sizes[0], kernel_size=12, stride=4, padding=2),   #55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                         #27
            
            nn.Conv2d(sizes[0], sizes[1], kernel_size=5, padding=2),       #27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                         #13
            
            nn.Conv2d(sizes[1], sizes[2], kernel_size=3, padding=1),       #13
            nn.ReLU(inplace=True),
            
            nn.Conv2d(sizes[2], sizes[3], kernel_size=3, padding=1),       #13
            nn.ReLU(inplace=True),
            
            nn.Conv2d(sizes[3], sizes[4], kernel_size=3, padding=1),       #13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                         #6
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(sizes[4] * 6 * 6, sizes[5]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(sizes[5], sizes[6]),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(sizes[6], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x    
    
    
