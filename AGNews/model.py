import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNet(nn.Module):

    def __init__(self,last_layer, WINDOW = 5, FEATURES = 1024):
        super(ConvNet, self).__init__()
        
        self.feat = FEATURES
        self.win = WINDOW
        self.last_layer = last_layer

        
        #self.pool1 = nn.MaxPool2d(self.pool_kernel_size)
        self.conv1 = nn.Conv2d(1, self.feat, (self.win, 300))
        
        #self.fc1 = nn.Linear(self.feat, 64)
        #self.dropout1 = nn.Dropout(0.35)
        #self.fc2 = nn.Linear(64,4)
        #self.dropout2 = nn.Dropout(0.35)
        
        self.fc1 = nn.Linear(self.feat, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32,4)
        self.dropout3 = nn.Dropout(0.2)
        
    
    def forward(self, x):
        
        x = self.conv1(x)
        #x = torch.sum(x, dim=3, keepdim=True)
        x = F.relu(x)

        x = F.max_pool2d(x, (x.shape[2],x.shape[3]))
        
        x = x.view(x.shape[0],-1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)

        if(self.last_layer):
            output = F.log_softmax(x, dim=1)
            return output
        else:
            return x
        
        
        
class ConvNet_Shallow(nn.Module):

    def __init__(self,last_layer, WINDOW = 3, FEATURES = 256):
        super(ConvNet_Shallow, self).__init__()
        
        self.feat = FEATURES
        self.win = WINDOW
        self.last_layer = last_layer

        
        #self.pool1 = nn.MaxPool2d(self.pool_kernel_size)
        self.conv1 = nn.Conv2d(1, self.feat, (self.win, 300))
        
        #self.fc1 = nn.Linear(self.feat, 64)
        #self.dropout1 = nn.Dropout(0.35)
        #self.fc2 = nn.Linear(64,4)
        #self.dropout2 = nn.Dropout(0.35)
        
        self.fc1 = nn.Linear(self.feat, 32)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32,4)
        self.dropout2 = nn.Dropout(0.2)
        
    
    def forward(self, x):
        
        x = self.conv1(x)
        #x = torch.sum(x, dim=3, keepdim=True)
        x = F.relu(x)

        x = F.max_pool2d(x, (x.shape[2],x.shape[3]))
        
        x = x.view(x.shape[0],-1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        
        if(self.last_layer):
            output = F.log_softmax(x, dim=1)
            return output
        else:
            return x

class ConvNet_Shallow_Single(nn.Module):

    def __init__(self,last_layer, WINDOW = 3, FEATURES = 256):
        super(ConvNet_Shallow_Single, self).__init__()

        self.feat = FEATURES
        self.win = WINDOW
        self.last_layer = last_layer


        #self.pool1 = nn.MaxPool2d(self.pool_kernel_size)
        self.conv1 = nn.Conv2d(1, self.feat, (self.win, 300))

        #self.fc1 = nn.Linear(self.feat, 64)
        #self.dropout1 = nn.Dropout(0.35)
        #self.fc2 = nn.Linear(64,4)
        #self.dropout2 = nn.Dropout(0.35)

        self.fc1 = nn.Linear(self.feat, 32)
        self.dropout1 = nn.Dropout(0.25)


    def forward(self, x):

        x = self.conv1(x)
        #x = torch.sum(x, dim=3, keepdim=True)
        x = F.relu(x)

        x = F.max_pool2d(x, (x.shape[2],x.shape[3]))

        x = x.view(x.shape[0],-1)

        x = self.fc1(x)
        x = self.dropout1(x)

        if(self.last_layer):
            output = F.log_softmax(x, dim=1)
            return output
        else:
            return x

