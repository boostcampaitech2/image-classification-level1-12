from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import PIL
import pandas as pd
import random
import tqdm
import torch

import sys
sys.path.append('/opt/ml/repos/project_notemplate_LJH')
from cust_util.util import mask_label_check, age_label_check, to_label

class MaskDataset(Dataset):
    def __init__(self, dir_list: list, mode, shuffle=True):
        self.gender_map = {'female': 1, 'male': 0}
        self.mode = mode
        self.dir_list = dir_list

        if shuffle == True:
            random.shuffle(self.dir_list)
        
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
        
        self.trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
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
        transforms.ColorJitter(brightness = 0.2, saturation = 0.2),
        transforms.ToTensor(), 
        transforms.RandomVerticalFlip(p = 0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #(0.1**0.5)*torch.randn(5, 10, 20): noise tensor
    img_list, labels = [], []
    for dir, label in batch:
        img = transform(PIL.Image.open(dir))

        img_list.append(img)
        labels.append(label)

    labels = torch.LongTensor(labels)
    return torch.stack(img_list, dim=0), labels


def test_collate_fn(batch):    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    img_list, labels = [], []
    for dir, label in batch:
        img = transform(PIL.Image.open(dir))
        img_list.append(img)
        labels.append(label)

    labels = torch.LongTensor(labels)
    return torch.stack(img_list, dim=0), labels
    
def sub_collate_fn(batch):    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_list = []
    for dir in batch:
        img_list.append(transform(PIL.Image.open(dir)))
    
    return torch.stack(img_list, dim=0)

##testcode##
##testcomplete --> output image가 좀 이상하다
# import sys
# sys.path.append('/opt/ml/project_LJH/')
# from cust_util.util import dirlister
# import os
# from PIL import Image
# import numpy as np
# import torch
# import time

# TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
# train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'))
# train_dir = dirlister(TRAIN_DATA_ROOT, meta = train_meta, mode = 'train')
# print(train_dir[0])

# dataset = MaskDataset(train_dir, 'train')
# dl = MaskDataLoader(dataset, 32, collate_fn=collate_fn)

# st = time.time()
# for idx, (img, age_label, gender_label, mask_label) in enumerate(dl):
#     if idx == 100:
#         break

# print(time.time()-st)
