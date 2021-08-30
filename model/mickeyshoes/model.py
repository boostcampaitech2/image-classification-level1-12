from torchvision import models as models
import torch.nn as nn

class MickeyMaskModel(nn.Module):
    """
    __init__(self, classes):
    classes : int, # class

    init_params(self):
    initialization parameters to all networks
    """
    def __init__(self, num_classes):
        super().__init__()
        self.models = models.resnet34(pretrained=True)
        self.models.fc = nn.Linear(512, num_classes)
        print(f'network # input channel : {self.models.conv1.weight.size(1)}')
        print(f'network # output channel : {self.models.fc.weight.shape[0]}')
        self.init_params()

    def forward(self, x):
        logit = self.models(x)
        return logit

    def init_params(self):
        for module in self.modules():
            # if isinstance(module, nn.Conv2d):
            #     #kaiming_normal
            #     nn.init.kaiming_normal_(module.weight)
            #     #BatchNorm2d
            # elif isinstance(module, nn.BatchNorm2d):
            #     nn.init.constant_(module.weight, 1)
            #     nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
