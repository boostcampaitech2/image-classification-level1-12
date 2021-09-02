from torchvision import transforms
from .custom_transform import SMOneClassTarget

mickeyshoes_training_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.CenterCrop((320,256)),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.Normalize(mean=(0.5601, 0.5241, 0.5014), std=(0.2331, 0.2430, 0.2456)),
])

mickeyshoes_valid_transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # transforms.CenterCrop((320,256)),
    # transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.Normalize(mean=(0.5601, 0.5241, 0.5014), std=(0.2331, 0.2430, 0.2456)),
])

mickeyshoes_train_target_transform = transforms.Compose([
    SMOneClassTarget(),
    # transforms.ToTensor()
])