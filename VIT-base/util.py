
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold



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
    dirlister(rootdir, metadata, 'train')
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

