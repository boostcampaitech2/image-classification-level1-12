import os
from sklearn.model_selection import KFold
import pandas as pd
import torch



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
    label = mask_label_chedk('normal.jpg')
    label >> 2
    '''
    if age_label < 30:
        age_label = 0
    if age_label >= 30 and age_label < 60:
        age_label = 1
    if age_label >= 60:
        age_label = 2

    return age_label

def dirlister(root: str, meta: pd.DataFrame, mode)->list:
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


# def to_label(mask, age, gender)->int:
#     m = torch.argmax(mask, dim = 1)
#     a = torch.argmax(age, dim = 1)
#     g = torch.argmax(gender, dim = 1)

#     return (6*m +3*g + a).tolist()


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


##testcode
# import torch
# m = torch.tensor([[8e-1, 6e-2,2e-2]])
# a = torch.tensor([[0.35,0.29,-0.02]])
# g = torch.tensor([[0.58,0.28]])

# print(to_label(m, a, g))
