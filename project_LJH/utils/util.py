import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sklearn.model_selection import KFold
import os


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def mask_label_check(filename, schema = {'/incorrect': 0, '/mask': 1, '/normal': 2}):
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


def dirlister(root: str, meta: pd.DataFrame, mode = 'train')->list:
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

def to_label()->int:
    raise NotImplementedError

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

    


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


####testcode####
#print(mask_label_check('incorrect_mask.jpg'), 
#      mask_label_check('mask1.jpg'), 
#      mask_label_check('normal.jpg')
#       )

# TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
# train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'))
# train_dir = dirlister(TRAIN_DATA_ROOT, meta = train_meta)
# cv_train = CV(train_dir, 5)
# for train, valid, test in cv_train:
#     print(len(train), len(valid), len(test))

# print('done')