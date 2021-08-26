import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class ResNet(nn.Module): 
    """simple mobilenet_v2 class using torchvision.models.mobilenet_v2"""
    def __init__(self): 
        super(ResNet, self).__init__() 

        self.model = models.resnet18(pretrained = True)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256, True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace = True),
            nn.Linear(256, 128, True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace = True),
            nn.Linear(128, 18, True)
        )
    
    
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet()
    model = model.cuda()
    summary(model, ((3, 224, 224)))