from torchvision import models as models
import torch.nn as nn
import math

class MickeyMaskModel(nn.Module):
    """
    __init__(self, classes, version_number):
    classes : int, # class
    version_number : int, model version number(18,34, default=50)

    init_params(self):
    initialization parameters to all networks
    """
    def __init__(self, num_classes: int, version_number: int):
        super().__init__()
        self.models = None
        if version_number == 18:
            self.models = models.resnet18(pretrained=True)
        elif version_number == 34:
            self.models = models.resnet34(pretrained=True)
        else:
            self.models = models.resnet50(pretrained=True)

        self.models.fc = nn.Linear(512, num_classes)
        print(f'network # input channel : {self.models.conv1.weight.size(1)}')
        print(f'network # output channel : {self.models.fc.weight.shape[0]}')
        self.init_params()

    def forward(self, x):
        logit = self.models(x)
        return logit

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(self.models.fc.weight)
                stdv = 1.0 / math.sqrt(self.models.fc.weight.size(1))
                self.models.fc.bias.data.uniform_(-stdv, stdv)

class MickeyDenseNet(nn.Module):
    """
    __init__(self, num_classes):
    num_classes: int, # class
    """

    def __init__(self, num_classes):
        super().__init__()
        self.models = models.densenet121(pretrained=True)
        self.models.classifier = nn.Linear(in_features=1024, out_features=num_classes)
        nn.init.xavier_uniform_(self.models.classifier.weight)
        stdv = 1.0 / math.sqrt(self.models.classifier.weight.size(1))
        self.models.classifier.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        logits = self.models(input)
        return logits