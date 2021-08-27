from torch import torch, nn
from torch import Tensor
from torchsummary import summary
import torchvision.transforms as T
from pytorch_pretrained_vit import ViT

class vision_transformer_for_age(nn.Module):
    """ Vision Transformer model
    classifying input images (x) into classes
    """
    def __init__(self, n_classes:int = 2):
        '''
        60대 이하, 60대 이상만을 학습시킨 baseline model
        '''
        super().__init__()

        self.backbone = ViT('B_16_imagenet1k', pretrained=True)
        self.backbone.fc = nn.Linear(768, n_classes, bias = True)

    def forward(self, x):
        x = self.backbone(x)
        return x


class vision_transformer(nn.Module):
    """ Vision Transformer model
    classifying input images (x) into classes
    """
    def __init__(self, n_classes:int = 18):
        super().__init__()
        pretrained_dir = ''
        self.age_back = vision_transformer_for_age().load_state_dict(pretrained_dir)
        self.age_back.backbone.fc = nn.Linear(768, n_classes, bias = True)

    def forward(self, x):
        x = self.backbone(x)
        return x