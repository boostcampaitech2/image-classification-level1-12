from torchvision.models import resnet18
from model.model import resnet_finetune

my_model = resnet_finetune(resnet18, 18)

print(my_model)