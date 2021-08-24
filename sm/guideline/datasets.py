"""
마스크 데이터셋을 읽고 전처리를 진행한 후 데이터를 하나씩 꺼내주는 Dataset 클래스를 구현한 파일입니다.

이 곳에서, 나만의 Data Augmentation 기법 들을 구현하여 사용할 수 있습니다.
"""
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import tqdm

get_datas = pd.read_csv(r'/opt/ml/data/train/train.csv')

print(get_datas)
