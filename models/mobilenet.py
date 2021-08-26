import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class MobileNet(nn.Module): 
    """simple mobilenet_v2 class using torchvision.models.mobilenet_v2"""
    def __init__(self): 
        super(MobileNet, self).__init__() 

        self.features = models.mobilenet_v2(pretrained=True).features
    
        
        self.gap = nn.AdaptiveAvgPool2d(output_size = (1, 1))

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p = 0.2)
        
        self.fc1 = nn.Linear(in_features = 1280, out_features = 256, bias = True)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(in_features = 256, out_features = 18, bias = True)

    
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(-1, 1280)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = MobileNet()
    model = model.cuda()
    summary(model, ((3, 512, 384)))
