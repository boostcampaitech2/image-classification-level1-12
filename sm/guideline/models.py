from torchvision import models as models
import torch.nn as nn
import math


class MaskModel(nn.Module):
    """
    __init__(self, classes):
    classes : int, # class

    init_params(self):
    initialization parameters to all networks
    """

    def __init__(self, classes):
        super().__init__()
        self.models = models.resnet34(pretrained=True)
        self.models.fc = nn.Linear(512, classes)
        print(f"network # input channel : {self.models.conv1.weight.size(1)}")
        print(f"network # output channel : {self.models.fc.weight.shape[0]}")
        nn.init.xavier_uniform_(self.models.fc.weight)
        stdv = 1.0 / math.sqrt(self.models.fc.weight.size(1))
        self.models.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        logit = self.models(x)
        return logit


class MickeyDenseNet(nn.Module):
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
