import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import StratifiedKFold

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
        dirname = Path(os.path.join(self.dirname, df_name))
        if not dirname.is_dir():
            dirname.mkdir(parents=True, exist_ok=False)

        for idx in range(len(df)):
            path = df["path"][idx]
            name = df["name"][idx]
            shutil.copy(path, os.path.join(dirname, name))

    def train_val_split(self, train_label):
        KFold = StratifiedKFold(5)
        X = train_label.index
        y = train_label["label"]

        for idx, (train_idx, val_idx) in enumerate(KFold.split(X, y)):
            globals()[f"train_df{idx}"] = train_label[
                train_label.index.isin(X[train_idx])
            ].reset_index(drop=True)
            globals()[f"val_df{idx}"] = train_label[
                train_label.index.isin(X[val_idx])
            ].reset_index(drop=True)

        train_list = [train_df0, train_df1, train_df2, train_df3, train_df4]
        val_list = [val_df0, val_df1, val_df2, val_df3, val_df4]

        return train_list, val_list


if __name__ == "__main__":
    train_path = "/opt/ml/image-classification-level1-12/templates/data/train"
    train_label = pd.read_csv(os.path.join(train_path, "train_with_label.csv"))
    run_train = Run_Split(os.path.join(train_path, "image_all"))
    train_list, val_list = run_train.train_val_split(train_label)
    for i in tqdm.tqdm(range(5)):
        run_train.image_all(train_list[i], f"train{i}_image")
        run_train.image_all(val_list[i], f"val{i}_image")
