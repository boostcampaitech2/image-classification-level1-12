import os
from copy import deepcopy
from collections import defaultdict
import random
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class MaskDataset(VisionDataset):

    TRAIN_IMAGE_TYPES = ("normal", "mask1", "mask2", "mask3", "mask4", "mask5", "incorrect_mask")
    TRAIN_FEATURES = ("id", "gender", "age_class", "image_path")
    GENDERS = ("male", "female")
    AGE_CLASSES = ("<30", ">=30 and <60", ">=60")
    MASK_CLASSES = ("mask", "incorrect_mask", "normal")
    IMAGE_FILE_EXTENSION = (".jpg", ".jpeg", ".png")

    def __init__(
            self,
            root: str,
            train: bool = True,
            num_folds: int = 1,
            folds: Optional[List[int]] = None,
            random_seed: int = 86,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ):
        """
        :param root: the absolute path of data directory
        :param train: train data를 원하는 경우 True, eval data를 원하는 경우 False
        :param num_folds: the number of folds
        :param folds: fold numbers to select, 0 ~ num_folds -1
        :param random_seed: seed for shuffle
        :param transform: transform to be applied to image
        """
        super(MaskDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.num_folds = num_folds
        self.folds = folds
        if not self.folds:
            folds = list(range(self.num_folds))
        self.random_seed = random_seed

        self.selected_data = list()

        if self.train:
            # read csv and shuffle
            meta_data_all = pd.read_csv(os.path.join(root + "/train/train.csv"))
            np.random.seed(self.random_seed)
            # 위 seed 설정은 아래 shuffle 직전에 항상 해주어야 한다.
            # Dataset이 어떠한 상황에서도 (여러번 옵션이 바뀌며 생성 되어도) split이 항상 동일하게 되게 하기 위함이다.
            meta_data_all = meta_data_all.sample(frac=1).reset_index(drop=True)

            # convert "gender", "age"
            meta_data_all["gender"] = meta_data_all["gender"].map(self._gender_to_gender_class)
            # meta_data_all["age"] = meta_data_all["age"].map(self._age_to_age_class)
            meta_data_all["age"] = meta_data_all["age"].map(self._age_to_age)

            # collect by class(gender, age)
            class_to_data_dict = defaultdict(list)
            for _, meta_row in meta_data_all.iterrows():
                class_to_data_dict[(meta_row["gender"], meta_row["age"])].append(meta_row)

            # split each class data and select folds
            for fold in folds:
                for a_class, class_data in class_to_data_dict.items():
                    start_idx = len(class_data) * fold // self.num_folds
                    end_idx = len(class_data) * (fold + 1) // self.num_folds
                    self.selected_data.extend(class_data[start_idx:end_idx])
            random.shuffle(self.selected_data)
        else:
            # read csv
            self.selected_data = \
                [meta_row for _, meta_row in pd.read_csv(os.path.join(root + "/eval/info.csv")).iterrows()]
            # DO NOT shuffle eval data
            # np.random.seed(self.random_seed)
            # self.selected_data = self.selected_data.sample(frac=1).reset_index(drop=True)

        # make the full paths of images and save them with target infos to the list
        self.image_path_and_info_list = list()
        for a_data in self.selected_data:
            if self.train:
                train_infos_except_mask = \
                    {key: value for key, value in a_data.items() if key in ["id", "gender", "age", "race"]}

                for image_type in self.TRAIN_IMAGE_TYPES:
                    image_path = os.path.join(root + "/train/images/" + a_data["path"] + "/" + image_type)

                    train_info = deepcopy(train_infos_except_mask)
                    train_info["mask"] = self._mask_to_mask_class(image_type)

                    self.image_path_and_info_list.append((image_path, train_info))
            else:
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
        if train
            image path
                /opt/ml/mask_data/train/images/001726_male_Asian_26/mask3.jpg
            target
                (
                    0  # MASK_CLASSES = ("mask", "incorrect_mask", "normal")
                    0,  # GENDERS = ("male", "female")
                    60  # age
                )
        if not train (eval)
            image path
                /opt/ml/mask_data/eval/images/9da5ae3d63373e1e44e9a323d17779f2b661e50d (file extension is not included)
            info
                0
        """
        img_path, info = self.image_path_and_info_list[idx]

        # load and transform image
        img = None
        if self.train:
            for image_file_extension in self.IMAGE_FILE_EXTENSION:
                try:
                    img = Image.open(img_path + image_file_extension)
                    break
                except FileNotFoundError:
                    continue
        else:
            img = Image.open(img_path)

        if not img:
            raise FileNotFoundError(f'No such file: {img_path}{" or ".join(self.IMAGE_FILE_EXTENSION)}')

        if self.transform is not None:
            img = self.transform(img)

        # target
        if self.train:
            target = (info["mask"], info["gender"], info["age"])
        else:
            target = info

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @classmethod
    def _age_to_age_class(cls, age):
        int_age = int(age)
        if int_age < 30:
            return 0
        elif 30 <= int_age < 60:
            return 1
        else:  # int_age >= 60
            return 2

    @classmethod
    def _age_to_age(cls, age):
        int_age = int(age)
        if int_age < 30:
            return int_age
        elif 30 <= int_age < 60:
            return int_age
        else:  # int_age >= 60
            return age + 20

    @classmethod
    def _gender_to_gender_class(cls, gender):
        if gender == "male":
            return 0
        elif gender == "female":
            return 1
        else:
            raise ValueError(f'Gender-"{gender}" is not available in (cls.__class__.__name__).')

    @classmethod
    def _mask_to_mask_class(cls, mask_type):

        if mask_type.startswith("mask"):
            return 0
        elif mask_type == "incorrect_mask":
            return 1
        elif mask_type == "normal":
            return 2
        else:
            raise ValueError(f'Mask type - "{mask_type}" is not available.')
