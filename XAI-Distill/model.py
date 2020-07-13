from layers import *
import torch.nn as nn

class TeacherNet(nn.Module):
    def __init__(self,in_p=0,hidden_p=0):
        super(TeacherNet, self).__init__()
        self.fc1 = Linear(28 * 28, 1200)
        self.fc2 = Linear(1200, 1200)
        self.fc3 = Linear(1200, 10)
        
        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)
        
        self.d_in = Dropout(in_p)
        self.d_hidden1 = Dropout(hidden_p)
        self.d_hidden2 = Dropout(hidden_p)
        
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.d_in(x)
        x = self.d_hidden1(self.relu1(self.fc1(x)))
        x = self.d_hidden2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def RAP_relprop(self, R):
        
        R = self.fc3.RAP_relprop(R)
        R = self.d_hidden2.RAP_relprop(R)
        R = self.relu2.RAP_relprop(R)
        R = self.fc2.RAP_relprop(R)
        R = self.d_hidden1.RAP_relprop(R)
        R = self.relu1.RAP_relprop(R)
        R = self.fc1.RAP_relprop(R)
        R = self.d_in.RAP_relprop(R)
        R = R.reshape((-1,28,28))
        
        return R
    
    def relprop(self, R, alpha):
        
        R = self.fc3.relprop(R, alpha)
        R = self.d_hidden2.relprop(R, alpha)
        R = self.relu2.relprop(R, alpha)
        R = self.fc2.relprop(R, alpha)
        R = self.d_hidden1.relprop(R, alpha)
        R = self.relu1.relprop(R, alpha)
        R = self.fc1.relprop(R, alpha)
        R = self.d_in.relprop(R, alpha)
        R = R.reshape((-1,28,28))
        
        return R
    
    
class StudentNet(nn.Module):
    def __init__(self,in_p=0,hidden_p=0):
        super(StudentNet, self).__init__()
        
        self.fc1 = Linear(28 * 28, 800)
        self.fc2 = Linear(800, 10)
        
        self.relu = ReLU(inplace=True)
        
        self.d_in = Dropout(in_p)
        self.d_hidden = Dropout(hidden_p)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.d_in(x)
        x = self.d_hidden(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def RAP_relprop(self, R):
        
        R = self.fc2.RAP_relprop(R)
        R = self.d_hidden.RAP_relprop(R)
        R = self.relu.RAP_relprop(R)
        R = self.fc1.RAP_relprop(R)
        R = self.d_in.RAP_relprop(R)
        R = R.reshape((-1,28,28))
        
        return R
    

class KidNet(nn.Module):
    def __init__(self,in_p=0,hidden_p=0):
        super(KidNet, self).__init__()
        self.fc1 = Linear(28 * 28, 30)
        self.fc2 = Linear(30, 10)
        
        self.relu = ReLU(inplace=True)
        
        self.d_in = Dropout(in_p)
        self.d_hidden = Dropout(hidden_p)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.d_in(x)
        x = self.d_hidden(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def RAP_relprop(self, R):
        
        R = self.fc2.RAP_relprop(R)
        R = self.d_hidden.RAP_relprop(R)
        R = self.relu.RAP_relprop(R)
        R = self.fc1.RAP_relprop(R)
        R = self.d_in.RAP_relprop(R)
        R = R.reshape((-1,28,28))
        
        return R
    
