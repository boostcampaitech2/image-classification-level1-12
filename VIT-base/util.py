from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from torch import torch, nn
import os
import pandas as pd


class image_embedding(nn.Module):
  def __init__(self, in_channels: int = 3, img_size: int = 224, patch_size: int = 16, emb_dim: int = 16*16*3):
    super().__init__()

    self.rearrange = Rearrange('b c (num_w p1) (num_h p2) -> b (num_w num_h) (p1 p2 c) ', p1=patch_size, p2=patch_size)
    self.linear = nn.Linear(in_channels * patch_size * patch_size, emb_dim)

    self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
    
    n_patches = img_size * img_size // patch_size**2
    self.positions = nn.Parameter(torch.randn(n_patches + 1, emb_dim))

  def forward(self, x):
    batch, channel, width, height = x.shape

    x = self.rearrange(x) # flatten patches 
    x = self.linear(x) # embedded patches 

    # ================ ToDo1 ================ #
    # (1) Build [token; image embedding] by concatenating class token with image embedding
    c = repeat(self.cls_token, '() n d -> b n d', b=batch) 
    x = torch.cat((c ,x), axis = 1)

    # (2) Add positional embedding to [token; image embedding]
    x = x + self.positions
    # ======================================= #
    return x


def mask_label_check(filename, schema = {'/incorrect': 1, '/mask': 0, '/normal': 2})->int:
    '''
    usage
    label = mask_label_chedk('normal.jpg')
    label >> 2
    '''
    for k in schema.keys():
        if k in '/' + filename:
            return schema[k]

    print('unexpected label detected. check schema')
    raise Exception


def age_label_check(age_label)->int:
    '''
    usage
    mask_label_chedk('normal.jpg')
    >> 2
    '''
    if age_label < 30:
        age_label = 0
    if age_label >= 30 and age_label < 60:
        age_label = 1
    if age_label >= 60:
        age_label = 2

    return age_label

def dirlister(root: str, meta: pd.DataFrame, mode)->list:
    '''
    returns every image path which is in form

    usage
    dirlister(rootdir, metadata)
    >>> [1.jpg, 2.jpg.....]
    '''
    mode_dict = {'train': 'path', 'sub':'ImageID'}
    image_dirs = [os.path.join(root,'images', x) for x in meta[mode_dict[mode]].values]
    if mode == 'train':
        image_path = []
        for dir in image_dirs:
            for filenames in os.listdir(dir):
                tmp = os.path.join(dir, filenames)
                if os.path.isfile(tmp) and '._' not in tmp:
                    image_path.append(tmp)
                    
    else:
        return image_dirs
        
    return image_path


def to_label(mask, age, gender)->int:
    if int(6*mask +3*gender + age) > 17:
        raise Exception
    return int(6*mask +3*gender + age)

class CV(object):
    def __init__(self, dirs: list, fold_num: int, sort = True):
        self.current = 0
        self.maxfold = fold_num

        self.kf = KFold(n_splits = fold_num)
        self.fold_index = []  #list of train_valid_test

        if sort:
            self.dirs = sorted(dirs)
        else: 
            self.dirs = dirs

        for train_index, test_index in self.kf.split(dirs):
            split = len(test_index)//2 - 1
            valid_index = test_index[:split]
            test_index = test_index[split:]
            self.fold_index.append([train_index, valid_index, test_index])


    def __iter__(self):
        return self


    def __next__(self):
        #gives next fold
        if self.current >= self.maxfold:
            raise StopIteration
        else:
            train_id, valid_id, test_id = self.fold_index[self.current]
            train = [self.dirs[idx] for idx in train_id]
            valid = [self.dirs[idx] for idx in valid_id]
            test = [self.dirs[idx] for idx in test_id]
            self.current += 1

            return [train, valid, test]