from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import PIL
import pandas as pd
import random
from torchvision.transforms.transforms import CenterCrop
import tqdm
import torch

import sys
sys.path.append('/opt/ml/repos/project_notemplate_LJH')
from util import mask_label_check, age_label_check, to_label

class MaskDataset(Dataset):
    def __init__(self, dir_list: list, mode, shuffle=True):
        self.gender_map = {'female': 1, 'male': 0}
        self.mode = mode
        self.dir_list = dir_list

        if shuffle == True:
            random.shuffle(dir_list)
        
        self.label = []
        for dir in tqdm.tqdm(self.dir_list, leave=False, desc = '    creating dataset: '):
            split_dir = dir.split('/')
            folder_name = split_dir[-2]
            file_name = split_dir[-1]
            if self.mode == 'train':
                age = int(folder_name.split('_')[-1])
                gender = folder_name.split('_')[-3]
                
                gender_label = self.gender_map[gender]
                mask_label = mask_label_check(file_name)
                age_label = age_label_check(age)

                self.label.append(to_label(mask_label, age_label, gender_label))


    def __len__(self):
        return len(self.dir_list)


    def __getitem__(self, index):
        if self.mode == 'train':
            return self.dir_list[index], self.label[index]
        else:
            return self.dir_list[index]



class MaskDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=1, collate_fn = None):
        
        self.mode = dataset.mode
        self.length = len(dataset)
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)

    def __len__(self):
        return self.length
        
        
def collate_fn(batch):    
    transform = transforms.Compose([
        transforms.CenterCrop(384),
        transforms.ToTensor(), 
        transforms.RandomVerticalFlip(p = 0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #(0.1**0.5)*torch.randn(5, 10, 20): noise tensor
    img_list, labels = [], []
    for dir, label in batch:
        img = transform(PIL.Image.open(dir))
        img = img + (0.1**0.5)*torch.randn(img.shape)
        img_list.append(img)
        labels.append(label)

    labels = torch.LongTensor(labels)
    return torch.stack(img_list, dim=0), labels


def test_collate_fn(batch):    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.CenterCrop(384),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    img_list, labels = [], []
    for dir, label in batch:
        img = transform(PIL.Image.open(dir))
        img_list.append(img)
        labels.append(label)

    labels = torch.LongTensor(labels)
    return torch.stack(img_list, dim=0), labels
    
def sub_collate_fn(batch):    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.CenterCrop(384),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_list = []
    for dir in batch:
        img_list.append(transform(PIL.Image.open(dir)))
    
    return torch.stack(img_list, dim=0)


class AgeDataset(Dataset):
    def __init__(self, dir_list: list, label_list: list, shuffle=True):
        self.dir_list = dir_list
        self.label = label_list

        if shuffle == True:
            print('shuffle mode')


    def __len__(self):
        return len(self.dir_list)


    def __getitem__(self, index):
        return self.dir_list[index], self.label[index]



class AgeDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=1, collate_fn = None):
        
        self.length = len(dataset)
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)

    def __len__(self):
        return self.length