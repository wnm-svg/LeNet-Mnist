# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4x4 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

      
        

        # self.conv1 = nn.Conv2d(1, 3, 5)
        # self.conv2 = nn.Conv2d(3, 8, 5)
        # self.fc1 = nn.Linear(128, 60)  # 4x4 image dimension
        # self.fc2 = nn.Linear(60, 42)
        # self.fc3 = nn.Linear(42, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)   
        return x


class Net_S(nn.Module):
    def __init__(self):
        super(Net_S, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.quant1 = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4x4 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dequant1 = torch.quantization.DeQuantStub()
      
        

        # self.conv1 = nn.Conv2d(1, 3, 5)
        # self.conv2 = nn.Conv2d(3, 8, 5)
        # self.fc1 = nn.Linear(128, 60)  # 4x4 image dimension
        # self.fc2 = nn.Linear(60, 42)
        # self.fc3 = nn.Linear(42, 10)

    def forward(self, x):
        x = self.quant1(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.contiguous().view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant1(x)
        return x
