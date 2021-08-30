import math

import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision
from torchvision.models.resnet import resnet34

class FineTuneResNet_HD(nn.Module):
    """
    resnet model명과 output class 개수를 입력해주면
    그것에 맞는 모델을 반환, pretrained, bias initialize
    """
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=256, bias=True)
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p = 0.2)
        self.fc2 = nn.Linear(in_features=256 , out_features= 18, bias=True)
        
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1.0 / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        
        x= self.model(x)
        x= self.relu(x)
        x= self.dropout(x)
        x= self.fc2(x)
        
        return x


class FineTuneResNet18(nn.Module):
    """
    resnet model명과 output class 개수를 입력해주면
    그것에 맞는 모델을 반환, pretrained, bias initialize
    """
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=18, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1.0 / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        return self.model(x)