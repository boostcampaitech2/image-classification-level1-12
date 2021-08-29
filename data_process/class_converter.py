from typing import Tuple

import torch


def age_to_age_class(age: str) -> int:
    int_age = int(age)
    if int_age < 30:
        return 0
    elif 30 <= int_age < 60:
        return 1
    else:  # int_age >= 60
        return 2


def gender_to_gender_class(gender: str) -> int:
    if gender == "male":
        return 0
    elif gender == "female":
        return 1
    else:
        raise ValueError(f'Gender-"{gender}" is not available in (cls.__class__.__name__).')


def mask_to_mask_class(mask_type: str) -> int:
    if mask_type.startswith("mask"):
        return 0
    elif mask_type == "incorrect_mask":
        return 1
    elif mask_type == "normal":
        return 2
    else:
        raise ValueError(f'Mask type - "{mask_type}" is not available.')


def convert_3class_to_1class(mask_class: int, gender_class: int, age_class: int) -> int:
    return mask_class * 6 + gender_class * 3 + age_class


def convert_3class_to_1class_batch_tensor(output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mask_pred = torch.argmax(output[0], dim=1)
    gender_pred = torch.argmax(output[1], dim=1)
    age_pred = torch.argmax(output[2], dim=1)

    return torch.mul(mask_pred, 6) + torch.mul(gender_pred, 3) + age_pred