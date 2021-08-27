import torch
from Model import MaskModel
from torchvision import datasets, transforms
import PIL

model = MaskModel()
#model.load_state_dict(torch.load('/opt/ml/repos/project_notemplate_LJH/results/weight.pt'))
transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dir = 'input/data/eval/images/248428d9a4a5b6229a7081c32851b90cb8d38d0c.jpg'
img = transform(PIL.Image.open(dir))
img = torch.stack([img])
age, gender, mask = model(img)

print(f'age: {age}')
print(f'gender: {gender}')
print(f'mask: {mask}')