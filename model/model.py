import math

import torch
import torch.nn as nn
from torchvision.models import resnet18


class FineTuneResNet(nn.Module):
    """
    resnet model명과 output class 개수를 입력해주면
    그것에 맞는 모델을 반환, pretrained, bias initialize
    """
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1.0 / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)

    def __call__(self, x):
        return self.model(x)