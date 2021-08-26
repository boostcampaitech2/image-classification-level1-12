import os
import torch
import pandas as pd
import numpy
from  torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import time
from PIL import Image


class ImageDataset(Dataset):
    """Image dataset by group and classes using torchvision.datasets.ImageFolder"""
    
    def __init__(self, image_path:str, transforms = None):
        self.transforms = transforms
        self.data = ImageFolder(image_path, transforms)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx][0], self.data[idx][1]
        return img, target


class TestDataset(Dataset):
    def __init__(self, df, path, transforms):
        self.df = df
        self.path = path
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        x = Image.open(self.path + "/" + self.df.iloc[index]['ImageID'])
        if self.transforms:
            x = self.transforms(x)
        return x


# https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/2
if __name__ == "__main__":
    
    base_dir = "../data/train/"
    group1_dir = "train1"
    
    transform_base = transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]) 

    group1_dataset = ImageDataset(base_dir + group1_dir, transforms = transform_base)
    group1_loader = DataLoader(group1_dataset, batch_size = 64, shuffle = True, num_workers = 8)
    
    start = time.time()
    for idx, (x, y) in enumerate(group1_loader):
        if idx == 100:
            break
    print(time.time() - start)
    
