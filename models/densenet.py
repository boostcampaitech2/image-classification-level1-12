import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class DenseNet(nn.Module): 
    """simple mobilenet_v2 class using torchvision.models.mobilenet_v2"""
    def __init__(self): 
        super(DenseNet, self).__init__() 

        self.features = models.densenet201(pretrained = True).features
        
        self.gap = nn.AdaptiveAvgPool2d(output_size = (1, 1))

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features = 1920, out_features = 512, bias = True)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(in_features = 512, out_features = 256, bias = True)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(in_features = 256, out_features = 18, bias = True)
    
    
    def forward(self, x):
        x = self.features(x)

        x = self.gap(x)
        x = x.view(-1, 1920)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DenseNet()
    model = model.cuda()
    y = model(torch.randn(64, 3, 224, 224).cuda())
    print(y.size())