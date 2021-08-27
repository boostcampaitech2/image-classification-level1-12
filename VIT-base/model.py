from torch import torch, nn
from torch import Tensor
from torchsummary import summary
import torchvision.transforms as T
from pytorch_pretrained_vit import ViT


class vision_transformer(nn.Module):
    """ Vision Transformer model
    classifying input images (x) into classes
    """
    def __init__(self, in_channel: int = 3, img_size:int = 224, 
                 patch_size: int = 16, emb_dim:int = 16*16*3, 
                 n_enc_layers:int = 15, num_heads:int = 3, 
                 forward_dim:int = 4, dropout_ratio: float = 0.2, 
                 n_classes:int = 18):
        super().__init__()

        model = ViT('B_16_imagenet1k', pretrained=True)
        model.fc = nn.Linear(768, 18, bias = True)

    def forward(self, x):
        x = self.model(x)
        return x