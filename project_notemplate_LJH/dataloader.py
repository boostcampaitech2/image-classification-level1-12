from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import PIL
import pandas as pd
import random
import tqdm
import torch

import sys
sys.path.append('/opt/ml/repos/project_notemplate_LJH')
from cust_util.util import mask_label_check

class MaskDataset(Dataset):
    def __init__(self, dir_list: list, meta: pd.DataFrame, shuffle=True, validation_split=0.0, num_workers=1, training=True, mode = 'train'):
        
        modes = {'train': 'path', 'test': 'eval'}
        gender = {'female': 0, 'male': 1}
        self.mode = mode
        self.dir_list = dir_list
        self.meta = meta
        if shuffle == True:
            random.shuffle(self.dir_list)
        
        self.label = []

        for dir in tqdm.tqdm(self.dir_list):
            split_dir = dir.split('/')
            key = split_dir[-2]
            if self.mode == 'train':
                age_label, gender_label = self.meta[self.meta['path'] == key][['age', 'gender']].values[0]
                gender_label = gender[gender_label]
                mask_label = mask_label_check(split_dir[-1])

                self.label.append([age_label, gender_label, mask_label])




    def __len__(self):
        return len(self.dir_list)


    def __getitem__(self, index):
        if self.mode == 'train':
            return self.dir_list[index], self.label[index]
        else:
            return self.dir_list[index]



class MaskDataLoader(DataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, batch_size, num_workers=1, collate_fn = None, validataion_split = 0.0):
        self.trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.mode = dataset.mode

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        super().__init__(**self.init_kwargs)
        
        
def collate_fn(batch):    
    transform = transforms.Compose([transforms.ToTensor()])
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])

    data_list, age_list, gender_list, mask_list = [], [], [], []

    for dir, (age_label, gender_label, mask_label) in batch:
        data_list.append(transform(PIL.Image.open(dir)))

        if age_label < 30:
            age_label = 0
        elif age_label < 60:
            age_label = 1
        else:
            age_label = 2
            
        age_list.append(age_label)
        gender_list.append(gender_label)
        mask_list.append(mask_label)
    
    
    return torch.stack(data_list, dim=0), torch.LongTensor(age_list), torch.LongTensor(gender_list), torch.LongTensor(mask_list)
    
    
def sub_collate_fn(batch):    
    transform = transforms.Compose([transforms.ToTensor()])
    data_list = []

    for dir in batch:
        data_list.append(transform(PIL.Image.open(dir)))
    
    return torch.stack(data_list, dim=0)

##testcode##
##testcomplete --> output image가 좀 이상하다
# import sys
# sys.path.append('/opt/ml/project_LJH/')
# from utils.util import dirlister
# import os
# from PIL import Image
# import numpy as np
# import torch
# import time

# TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
# train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'))
# train_dir = dirlister(TRAIN_DATA_ROOT, meta = train_meta)
# print(train_dir[0])

# dataset = MaskDataset(train_dir, train_meta)
# dl = MaskDataLoader(dataset, 32, collate_fn=collate_fn)

# st = time.time()
# for idx, (img, age_label, gender_label, mask_label) in enumerate(dl):
#     if idx == 100:
#         break

# print(time.time()-st)
