import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class MaskModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        ###basemodel###
        self.backbone = models.densenet161(pretrained=True)
        self.backbone.conv0 = nn.Conv2d(3, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.backbone.classifier = nn.Linear(in_features=2208, out_features=1500, bias=True)
        self.fc1 = nn.Linear(1500, 300)

        self.fc_age = nn.Linear(300, num_classes)
        self.fc_gender = nn.Linear(300, 2)
        self.fc_mask = nn.Linear(300, num_classes)

    def forward(self, x):
        
        #backbone
        x = self.backbone(x)
        x = self.fc1(x)

        x_age = self.fc_age(x)
        x_gender = self.fc_gender(x)
        x_mask = self.fc_mask(x)

        return x_age, x_gender, x_mask 



