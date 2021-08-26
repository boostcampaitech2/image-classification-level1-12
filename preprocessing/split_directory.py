import pandas as pd
import os
import shutil
from sklearn.model_selection import StratifiedKFold


def age_pre(x:int):
    if x < 30:
        return 0
    elif 30 <= x < 60:
        return 1
    elif x >= 60:
        return 2


def div_class(dir:str):
    gender = dir.split("/")[-2].split("_")[1]
    age = int(dir.split("/")[-2].split("_")[3])
    if "mask1" in dir or "mask2" in dir or "mask3" in dir or "mask4" in dir or "mask5" in dir:
        if gender == "male":
            if age < 30:
                return 0
            elif 30 <= age < 60:
                return 1
            elif age >= 60:
                return 2
        elif gender == "female":
            if age < 30:
                return 3
            elif 30 <= age < 60:
                return 4
            elif age >= 60:
                return 5
    elif "incorrect" in dir:
        if gender == "male":
            if age < 30:
                return 6
            elif 30 <= age < 60:
                return 7
            elif age >= 60:
                return 8
        elif gender == "female":
            if age < 30:
                return 9
            elif 30 <= age < 60:
                return 10
            elif age >= 60:
                return 11
    elif "normal" in dir:
        if gender == "male":
            if age < 30:
                return 12
            elif 30 <= age < 60:
                return 13
            elif age >= 60:
                return 14
        elif gender == "female":
            if age < 30:
                return 15
            elif 30 <= age < 60:
                return 16
            elif age >= 60:
                return 17


def age_pre(x:int):
    if x < 30:
        return 0
    elif 30 <= x < 60:
        return 1
    elif x >= 60:
        return 2


def gender_age_pre(df:pd.DataFrame):
    gender_age = []
    for i in range(len(df)):
        if df.iloc[i]['gender'] == "female":
            if df.iloc[i]['age_group'] == 0:
                gender_age.append(0)
            elif df.iloc[i]['age_group'] == 1:
                gender_age.append(1)
            elif df.iloc[i]['age_group'] == 2:
                gender_age.append(2)
        elif df.iloc[i]['gender'] == "male":
            if df.iloc[i]['age_group'] == 0:
                gender_age.append(3)
            elif df.iloc[i]['age_group'] == 1:
                gender_age.append(4)
            elif df.iloc[i]['age_group'] == 2:
                gender_age.append(5)
    df['gender_age'] = gender_age
    return df


def split_folder(dir_list:list, group_name:str, base_dir:str):
    for dir in dir_list:
        new_dir = base_dir + "images/" + dir
        person_dir = os.listdir(new_dir)
        person_dir = [p_d for p_d in person_dir if "._" not in p_d]
        for p_d in person_dir:
            old_p_dir = new_dir + "/" + p_d
            c = div_class(old_p_dir)
            new_p_dir = old_p_dir.split("/")[-2] + "_" + p_d
            if c == 0:
                shutil.copy(old_p_dir, base_dir + group_name + "/class00/" + new_p_dir)
            elif c == 1:
                shutil.copy(old_p_dir, base_dir + group_name + "/class01/" + new_p_dir)
            elif c == 2:
                shutil.copy(old_p_dir, base_dir + group_name + "/class02/" + new_p_dir)
            elif c == 3:
                shutil.copy(old_p_dir, base_dir + group_name + "/class03/" + new_p_dir)
            elif c == 4:
                shutil.copy(old_p_dir, base_dir + group_name + "/class04/" + new_p_dir)
            elif c == 5:
                shutil.copy(old_p_dir, base_dir + group_name + "/class05/" + new_p_dir)
            elif c == 6:
                shutil.copy(old_p_dir, base_dir + group_name + "/class06/" + new_p_dir)
            elif c == 7:
                shutil.copy(old_p_dir, base_dir + group_name + "/class07/" + new_p_dir)
            elif c == 8:
                shutil.copy(old_p_dir, base_dir + group_name + "/class08/" + new_p_dir)
            elif c == 9:
                shutil.copy(old_p_dir, base_dir + group_name + "/class09/" + new_p_dir)
            elif c == 10:
                shutil.copy(old_p_dir, base_dir + group_name + "/class10/" + new_p_dir)
            elif c == 11:
                shutil.copy(old_p_dir, base_dir + group_name + "/class11/" + new_p_dir)
            elif c == 12:
                shutil.copy(old_p_dir, base_dir + group_name + "/class12/" + new_p_dir)
            elif c == 13:
                shutil.copy(old_p_dir, base_dir + group_name + "/class13/" + new_p_dir)
            elif c == 14:
                shutil.copy(old_p_dir, base_dir + group_name + "/class14/" + new_p_dir)
            elif c == 15:
                shutil.copy(old_p_dir, base_dir + group_name + "/class15/" + new_p_dir)
            elif c == 16:
                shutil.copy(old_p_dir, base_dir + group_name + "/class16/" + new_p_dir)
            elif c == 17:
                shutil.copy(old_p_dir, base_dir + group_name + "/class17/" + new_p_dir)


def mkdir(base_dir:list, dir_list:list):
    for dir in base_dir:
        os.mkdir(dir)
    for dir1 in base_dir:
        for dir2 in dir_list:
            new_dir = dir1 + dir2
            os.mkdir(new_dir)
            

if __name__ == "__main__":
    train_dir = "../data/train/"
    meta_train = pd.read_csv(train_dir + "train.csv")
    meta_train['age_group'] = meta_train['age'].apply(age_pre)
    meta_train = gender_age_pre(meta_train)
    
    kfold = StratifiedKFold(n_splits = 5, shuffle = True)
    for idx, (train_idx, test_idx) in enumerate(kfold.split(meta_train, meta_train['gender_age'])):
        if idx == 0:
            df1 = meta_train.iloc[test_idx]
        elif idx == 1:
            df2 = meta_train.iloc[test_idx]
        elif idx == 2:
            df3 = meta_train.iloc[test_idx]
        elif idx == 3:
            df4 = meta_train.iloc[test_idx]
        elif idx == 4:
            df5 = meta_train.iloc[test_idx]
    
    mkdir(["../data/train/train1/", "../data/train/train2/", "../data/train/train3/", "../data/train/train4/", "../data/train/train5/"],
          ['class00', 'class01', 'class02', 'class03', 'class04', 'class05', 'class06', 'class07', 'class08', 'class09',
           'class10', 'class11', 'class12', 'class13', 'class14', 'class15', 'class16', 'class17'])
     
    mkdir(["../data/train/val1/", "../data/train/val2/", "../data/train/val3/", "../data/train/val4/", "../data/train/val5/"],
          ['class00', 'class01', 'class02', 'class03', 'class04', 'class05', 'class06', 'class07', 'class08', 'class09',
           'class10', 'class11', 'class12', 'class13', 'class14', 'class15', 'class16', 'class17'])
    
    mkdir(["../data/train/whole/"],
          ['class00', 'class01', 'class02', 'class03', 'class04', 'class05', 'class06', 'class07', 'class08', 'class09',
           'class10', 'class11', 'class12', 'class13', 'class14', 'class15', 'class16', 'class17'])
     

    train1 = pd.concat([df1, df2, df3, df4], axis = 0)
    val1 = df5

    train2 = pd.concat([df1, df2, df3, df5], axis = 0)
    val2 = df4

    train3 = pd.concat([df1, df2, df4, df5], axis = 0)
    val3 = df3

    train4 = pd.concat([df1, df3, df4, df5], axis = 0)
    val4 = df2

    train5 = pd.concat([df2, df3, df4, df5], axis = 0)
    val5 = df1

    split_folder(train1['path'], "train1", train_dir)
    split_folder(val1['path'], "val1", train_dir)  

    split_folder(train2['path'], "train2", train_dir)
    split_folder(val2['path'], "val2", train_dir) 

    split_folder(train3['path'], "train3", train_dir)
    split_folder(val3['path'], "val3", train_dir) 

    split_folder(train4['path'], "train4", train_dir)
    split_folder(val4['path'], "val4", train_dir) 

    split_folder(train5['path'], "train5", train_dir)
    split_folder(val5['path'], "val5", train_dir) 

    split_folder(meta_train['path'], "whole", train_dir)