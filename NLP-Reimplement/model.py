import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNet(nn.Module):

    def __init__(self,last_layer, WINDOW = 5, FEATURES = 1280):
        super(ConvNet, self).__init__()
        
        self.feat = FEATURES
        self.win = WINDOW
        self.last_layer = last_layer

        self.conv1 = nn.Conv2d(1, self.feat, (self.win, 300))
        
        #self.pool1 = nn.MaxPool2d(self.pool_kernel_size)
        
        self.fc1 = nn.Linear(self.feat, 160)
        self.dropout1 = nn.Dropout(0.35)
        self.fc2 = nn.Linear(160,20)
        self.dropout2 = nn.Dropout(0.35)


    def forward(self, x):
        
        #print(x.shape)
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (x.shape[2],x.shape[3]))
        
        #print(x.shape)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        if(self.last_layer):
            output = F.log_softmax(x, dim=1)
            return output
        else:
            return x