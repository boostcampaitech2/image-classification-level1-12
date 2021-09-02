from torch import torch, nn
from torchvision.models import resnet18
from torchsummary import summary
import torchvision.transforms as T
from pytorch_pretrained_vit import ViT
import math

# https://github.com/lukemelas/PyTorch-Pretrained-ViT
class LJH_vision_transformer(nn.Module):
    """ Vision Transformer model
    classifying input images (x) into classes
    """
    def __init__(self, num_classes:int = 18):
        super().__init__()
        self.backbone = ViT('L_32_imagenet1k', pretrained=False)
        self.backbone.fc = nn.Linear(1024, num_classes, bias = True)
        self.backbone.transformer.blocks[-1].drop = nn.Dropout(0.3)

        #add dropout
    def forward(self, x):
        x = self.backbone(x)
        return x


class LJH_FineTuneResNet(nn.Module):
    """
    resnet model명과 output class 개수를 입력해주면
    그것에 맞는 모델을 반환, pretrained, bias initialize
    """
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        #self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1.0 / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)