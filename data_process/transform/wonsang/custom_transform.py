from typing import Dict

import torch


def _age_to_age_class(age):
    int_age = int(age)
    if int_age < 30:
        return 0
    elif 30 <= int_age < 60:
        return 1
    else:  # int_age >= 60
        return 2


def _age_to_age_60plus(age, increment60):
    int_age = int(age)
    if int_age < 30:
        return int_age
    elif 30 <= int_age < 60:
        return int_age
    else:  # int_age >= 60
        return age + increment60


def _gender_to_gender_class(gender):
    if gender == "male":
        return 0
    elif gender == "female":
        return 1
    else:
        raise ValueError(f'Gender-"{gender}" is not available in (cls.__class__.__name__).')


def _mask_to_mask_class(mask_type):
    if mask_type.startswith("mask"):
        return 0
    elif mask_type == "incorrect_mask":
        return 1
    elif mask_type == "normal":
        return 2
    else:
        raise ValueError(f'Mask type - "{mask_type}" is not available.')


class TwoClassifierOneRegressionTargetTransform:
    def __init__(self, increment60: int = 20):
        self.increment60 = increment60

    def __call__(self, in_target: Dict):
        mask_class = _mask_to_mask_class(in_target["mask_file_name"])
        gender_class = _gender_to_gender_class(in_target["gender"])
        age = _age_to_age_60plus(in_target["age"], self.increment60)

        return torch.tensor((mask_class, gender_class, age), dtype=torch.float32)
