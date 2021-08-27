from torchvision import models as models
import torch.nn as nn

class MaskModel(nn.Module):
    """
    __init__(self, classes):
    classes : int, # class

    init_params(self):
    initialization parameters to all networks
    """
    def __init__(self, classes):
        super().__init__()
        self.models = models.resnet18(pretrained=True)
        self.models.fc = nn.Linear(512, classes)
        print(f'network # input channel : {self.models.conv1.weight.size(1)}')
        print(f'network # output channel : {self.models.fc.weight.shape[0]}')

    def forward(self, x):
        logit = self.models(x)
        return logit

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                #kaiming_normal 에 대한 설명 -> 뭐가 좋아서 쓴 것인지
                nn.init.kaiming_normal_(module.weight)
                #BatchNorm2d 가 무엇인지 설명
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)