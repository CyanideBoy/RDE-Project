import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNet(nn.Module):

    def __init__(self,WINDOW = 2, FEATURES = 800):
        super(ConvNet, self).__init__()
        
        self.win = WINDOW
        self.feat = FEATURES

        self.conv1 = nn.Conv2d(1, self.feat, (self.win,300))
        #print(self.pool_kernel_size)
        #self.pool1 = nn.MaxPool2d(self.pool_kernel_size)
        self.fc1 = nn.Linear(800, 20)
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2])
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (x.shape[2],x.shape[3]))
        #x = self.pool1(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.dropout1(x)
        output = F.log_softmax(x, dim=1)
        return output