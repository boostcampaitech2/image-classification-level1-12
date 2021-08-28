from torch import torch, nn
from torch import Tensor
from torchsummary import summary
import torchvision.transforms as T
from pytorch_pretrained_vit import ViT


class vision_transformer(nn.Module):
    """ Vision Transformer model
    classifying input images (x) into classes
    """
    def __init__(self, n_classes:int = 18):
        super().__init__()
        self.backbone = self.backbone = ViT('L_32_imagenet1k', pretrained=True)
        self.backbone.fc = nn.Linear(768, n_classes, bias = True)

    def forward(self, x):
        x = self.backbone(x)
        return x