from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import PIL
import pandas as pd
import random
import tqdm
import torch

class MaskDataset(Dataset):
    def __init__(self, dir_list: list, meta: pd.DataFrame, shuffle=True, validation_split=0.0, num_workers=1, training=True, mode = 'train'):
        
        modes = {'train': 'path', 'test': 'eval'}
        gender = {'female': 0, 'male': 1}
        self.mode = modes[mode]
        self.dir_list = dir_list
        self.meta = meta
        if shuffle == True:
            random.shuffle(self.dir_list)
        
        self.label = []

        for dir in tqdm.tqdm(self.dir_list):
            split_dir = dir.split('/')
            key = split_dir[-2]
            age_label, gender_label = self.meta[self.meta[self.mode] == key][['age', 'gender']].values[0]
            gender_label = gender[gender_label]
            mask_label = None
            if 'incorrect' in split_dir[-1]:
                mask_label = 0
            elif '' in split_dir[-1]:
                mask_label = 1
            elif '' in split_dir[-1]:
                mask_label = 1
            else:
                print('정의되지 않은 형식의 디렉토리입니다')
                raise Exception

            self.label.append([age_label, gender_label, mask_label])


    def __len__(self):
        return len(self.dir_list)


    def __getitem__(self, index):
            return self.dir_list[index], self.label[index]



class MaskDataLoader(DataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, batch_size, num_workers=1, collate_fn = None, validataion_split = 0.0):
        self.trsfm = transforms.Compose([
            transforms.ToTensor()
        ])

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        super().__init__(**self.init_kwargs)
        
        
def collate_fn(batch):    
    transform = transforms.Compose([transforms.ToTensor()])
    data_list, age_list, gender_list, mask_list = [], [], [], []

    for dir, (age_label, gender_label, mask_label) in batch:
        data_list.append(transform(PIL.Image.open(dir)))
        age_list.append(age_label)
        gender_list.append(gender_label)
        mask_list.append(mask_label)
    
    
    return torch.stack(data_list, dim=0), torch.LongTensor(age_list), torch.LongTensor(gender_list), torch.LongTensor(mask_list)
    


##testcode##
##testcomplete --> output image가 좀 이상하다
import sys
sys.path.append('/opt/ml/project_LJH/')
from utils.util import dirlister
import os
from PIL import Image
import numpy as np
import torch

TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'))
train_dir = dirlister(TRAIN_DATA_ROOT, meta = train_meta)[0:10]
print(train_dir[0])

dataset = MaskDataset(train_dir, train_meta)
dl = MaskDataLoader(dataset, 1, collate_fn=collate_fn)

t = next(iter(dl))
img, age_label, gender_label, mask_label = t
Image.fromarray(np.array(img[0].permute(1, 2, 0) * 255).astype(np.uint8)).convert('RGB').save('/opt/ml/project/test.jpg')

print(img[0].shape)







