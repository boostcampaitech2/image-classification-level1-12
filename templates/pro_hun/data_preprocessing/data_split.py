import os
import random
import shutil

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import StratifiedKFold
import argparse

from utils.util import ensure_dir

random_seed = 12
np.random.seed(random_seed)
random.seed(random_seed)


class Run_Split:
    def __init__(self, dirname):
        self.dirname = dirname

    def image_all(self, df: pd.DataFrame, df_name: str):
        """
        train dataset을 train, val에 맞춰서 폴더 재생성

        Args:
        dirname (str): '/opt/ml/image-classification-level1-12/templates/data/train/image_all'
        df (pd.DataFrame): train_df와 val_df
        df_name (str): 입력된 df에 맞춰서 입력
        """
        dirname = os.path.join(self.dirname, df_name)
        # 빈 directory 생성
        ensure_dir(dirname)

        for idx in range(len(df)):
            path = df["path"][idx]
            name = df["name"][idx]
            shutil.copy(path, os.path.join(dirname, name))

    def train_val_split(self, train_label, fold_num):
        KFold = StratifiedKFold(fold_num)
        X = train_label.index
        y = train_label["label"]

        train_list, val_list = [], []
        
        for train_idx, val_idx in KFold.split(X, y):
            train_list.append(train_label[train_label.index.isin(X[train_idx])].reset_index(drop=True))
            val_list.append(train_label[train_label.index.isin(X[val_idx])].reset_index(drop=True))


        return train_list, val_list


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--train_path', default="/opt/ml/image-classification-level1-12/templates/data/train", type=str, help='train_path')
    args.add_argument('--fold_num', default=5, type=int, help='kfold_num')
    
    args = args.parse_args()

    train_path = args.train_path
    train_label = pd.read_csv(os.path.join(train_path, "train_with_label.csv"))
    run_train = Run_Split(os.path.join(train_path, "image_all"))
    fold_num = args.fold_num
    train_list, val_list = run_train.train_val_split(train_label, fold_num)

    for i in tqdm.tqdm(range(fold_num)):
        run_train.image_all(train_list[i], f"train{i}_image")
        run_train.image_all(val_list[i], f"val{i}_image")
