import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class MaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        ###basemodel###
        self.backbone = models.densenet161(pretrained=True)
        self.name = 'densenet'
        self.backbone.classifier = nn.Linear(in_features=2208, out_features=18, bias=True)

    def forward(self, x):
        
        #backbone
        x = self.backbone(x)

        return x



