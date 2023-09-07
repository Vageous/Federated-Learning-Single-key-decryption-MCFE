from sys import modules
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.conv1=nn.Conv2d(1,6,kernel_size=5,stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,kernel_size=5,stride=1,padding=1)
        self.fc1=nn.Linear(16*6*6,120)
        # self.pool2=nn.MaxPool2d(2,2)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x=x.view(-1,16*6*6)#将数据平整为一维
        # print(x.shape)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class cnn1(nn.Module):
    def __init__(self) -> None:
        super(cnn1,self).__init__()
        self.conv1=nn.Conv2d(3,6,kernel_size=5,stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,kernel_size=5,stride=1,padding=1)
        self.fc1=nn.Linear(16*6*6,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*6*6)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


# # cnn=cnn()
# # print(cnn)
cn=cnn1()

# print(cn)