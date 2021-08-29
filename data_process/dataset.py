import os
from copy import deepcopy
from collections import defaultdict
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import PIL
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets.vision import VisionDataset

from data_process.class_converter import mask_to_mask_class, age_to_age_class


TRAIN_MASK_FILE_NAMES = ("normal", "mask1", "mask2", "mask3", "mask4", "mask5", "incorrect_mask")
TRAIN_FEATURES = ("id", "gender", "age_class", "image_path")
GENDERS = ("male", "female")
AGE_CLASSES = ("<30", ">=30 and <60", ">=60")
MASK_CLASSES = ("mask", "incorrect_mask", "normal")
IMAGE_FILE_EXTENSION = (
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
)


class MaskTrainDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            csv_path: str,
            num_folds: int = 5,
            person_shuffle: bool = True,
            set_num: int = 0,
            random_state: int = 74,
            training: bool = True
    ):
        """
        :param root: the absolute path of root image directory
        :param csv_path: the absolute path of csv file
        :param num_folds: the number of folds
        :param person_shuffle: train, valid 구분시 같은 사람의 여러 이미지가 나뉘어서 shuffle 될 수 있는지 여부
        :param set_num: (train-valid) set to select (0 ~ num_fold-1)
        :param random_state: random seed for KFold
        :param training: True - training data, False - validation data
        """
        super(MaskTrainDataset, self).__init__(root)
        self.csv_path = csv_path
        self.person_shuffle = person_shuffle
        self.num_folds = num_folds
        self.set_num = set_num
        self.random_state = random_state
        self.training = training

        # read csv and shuffle
        meta_data_all = pd.read_csv(csv_path)
        # 위 seed 설정은 아래 shuffle 직전에 항상 해주어야 한다.
        # Dataset이 어떠한 상황에서도 (여러번 옵션이 바뀌며 생성 되어도) split이 항상 동일하게 되게 하기 위함이다.
        meta_data_all = meta_data_all.sample(frac=1).reset_index(drop=True)

        # convert "gender", "age"
        # meta_data_all["gender"] = meta_data_all["gender"].map(gender_to_gender_class)
        # meta_data_all["age"] = meta_data_all["age"].map(age_to_age_class)

        # K-Fold
        # person_shuffle = True 이면
        #     "for mask_file_name in TRAIN_MASK_FILE_NAMES:" 가 K-Fold 전에 등장
        # person_shuffle = False 이면
        #     "for mask_file_name in TRAIN_MASK_FILE_NAMES:" 가 K-Fold 후에 등장
        selected_meta_data: List[Dict] = list()
        if self.num_folds < 2:
            for _, meta_row in meta_data_all.iterrows():
                for mask_file_name in TRAIN_MASK_FILE_NAMES:
                    data = {"mask_file_name": mask_file_name}
                    data.update(meta_row)
                    selected_meta_data.append(data)
        else:   # self.num_folds >= 2
            if self.person_shuffle:
                data_list: List[Dict] = list()
                class_list: List[str] = list()
                for _, meta_row in meta_data_all.iterrows():
                    for mask_file_name in TRAIN_MASK_FILE_NAMES:
                        data = {"mask_file_name": mask_file_name}
                        data.update(meta_row)
                        data_list.append(data)

                        class_tuple = (
                            str(mask_to_mask_class(mask_file_name)),
                            meta_row["gender"],
                            str(age_to_age_class(meta_row["age"]))
                        )
                        class_list.append("_".join(class_tuple))

                # split each class data and select folds
                skf = StratifiedKFold(n_splits=self.num_folds, random_state=self.random_state, shuffle=True)
                # self.num_folds 개의 set 중 self.set_num 번째를 택한다.
                train_set = list(skf.split(np.zeros(len(data_list)), class_list))[self.set_num]
                training_indexes, validation_indexes = train_set[0].tolist(), train_set[1].tolist()
                selected_indexes = training_indexes if self.training else validation_indexes
                selected_meta_data = [data_list[index] for index in selected_indexes]
            else:  # self.num_folds >= 2 and not self.person_shuffle:
                class_list: List[str] = list()
                for _, meta_row in meta_data_all.iterrows():
                    class_tuple = (
                        meta_row["gender"],
                        str(age_to_age_class(meta_row["age"]))
                    )
                    class_list.append("_".join(class_tuple))

                # split each class data and select folds
                skf = StratifiedKFold(n_splits=self.num_folds, random_state=self.random_state, shuffle=True)
                # self.num_folds 개의 set 중 self.set_num 번째를 택한다.
                train_set = list(skf.split(np.zeros(len(meta_data_all)), class_list))[self.set_num]
                training_indexes, validation_indexes = train_set[0].tolist(), train_set[1].tolist()
                selected_indexes = training_indexes if self.training else validation_indexes
                # meta data of selected people
                selected_people: List[Dict] = [meta_data_all[index] for index in selected_indexes]

                for meta_row in selected_people:
                    for mask_file_name in TRAIN_MASK_FILE_NAMES:
                        data = {"mask_file_name": mask_file_name}
                        data.update(meta_row)
                        selected_meta_data.append(data)

        # make the full paths(without extension) of images and save them with target infos to the list
        self.image_path_and_meta_list = list()
        for a_data in selected_meta_data:
            image_path = os.path.join(root + "/" + a_data["path"] + "/" + a_data["mask_file_name"])
            self.image_path_and_meta_list.append((image_path, a_data))


    def __len__(self):
        return len(self.image_path_and_meta_list)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        :param idx: image sequence number (not depending on meta data)
        :return: If transforms and target_transforms don't exist,
        image, meta_data of image (keys = csv file's columns + "mask_file_name") are returned.
        """
        img_path, target = self.image_path_and_meta_list[idx]

        # load and transform image
        img = None
        for image_file_extension in IMAGE_FILE_EXTENSION:
            try:
                img = PIL.Image.open(img_path + image_file_extension)
                break
            except FileNotFoundError:
                continue
        if not img:
            raise FileNotFoundError(f'No such file: {img_path}{" or ".join(IMAGE_FILE_EXTENSION)}')

        # image transform
        if self.transform is not None:
            img = self.transform(img)

        # target transform
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MaskEvalDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None
    ):
        """
        :param root: the absolute path of data directory
        :param transform: transform to be applied to image
        """
        super(MaskEvalDataset, self).__init__(root, transform=transform)

        # read csv
        self.selected_data = \
            [meta_row for _, meta_row in pd.read_csv(os.path.join(root + "/eval/info.csv")).iterrows()]

        # make the full paths of images and save them with target infos to the list
        self.image_path_and_info_list = list()
        for a_data in self.selected_data:
            image_path = os.path.join(root + "/eval/images/" + a_data["ImageID"])
            info = a_data["ans"]
            self.image_path_and_info_list.append((image_path, info))

    def __len__(self):
        return len(self.image_path_and_info_list)

    def __getitem__(self, idx):
        """
        :param idx: image sequence number (not depending on meta data)
        :return: (image path: str, label: Optional[Dict[str, Union[str,int]]])
        example
            image path
                /opt/ml/mask_data/eval/images/9da5ae3d63373e1e44e9a323d17779f2b661e50d (file extension is not included)
            info
                0
        """
        img_path, info = self.image_path_and_info_list[idx]

        # load and transform image
        img = PIL.Image.open(img_path)

        if not img:
            raise FileNotFoundError(f'No such file: {img_path}{" or ".join(IMAGE_FILE_EXTENSION)}')

        if self.transform is not None:
            img = self.transform(img)

        # target
        target = info

        return img, target

