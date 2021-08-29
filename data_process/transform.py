from torchvision import transforms

from data_process.custom_transform_class import AddGaussianNoise, OneClassTarget


baseline_training_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5601, 0.5241, 0.5014), std=(0.2331, 0.2430, 0.2456))
])

baseline_valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5601, 0.5241, 0.5014), std=(0.2331, 0.2430, 0.2456))
])

baseline_train_target_transform = transforms.Compose([
    OneClassTarget(),
    # transforms.ToTensor()
])

gaussian_transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0., 1.)
])


