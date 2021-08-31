from torchvision import transforms
from data_process.transform.baseline.custom_transform import AddGaussianNoise, OneClassTarget


LJH_training_transform = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5601, 0.5241, 0.5014), std=(0.2331, 0.2430, 0.2456)),
    #transforms.Grayscale(),
    AddGaussianNoise()
])

LJH_testing_transform = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5601, 0.5241, 0.5014), std=(0.2331, 0.2430, 0.2456))
    #transforms.Grayscale(),
])