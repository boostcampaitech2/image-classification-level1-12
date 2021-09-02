import torch

from data_process.class_converter import sm_mask_to_mask_class, gender_to_gender_class, age_to_age_class, convert_3class_to_1class

class SMOneClassTarget:
    def __call__(self, meta_data: dict) -> int:
        mask_class = sm_mask_to_mask_class(meta_data["mask_file_name"])
        gender_class = gender_to_gender_class(meta_data["gender"])
        age_class = age_to_age_class(meta_data["age"])

        total_class = convert_3class_to_1class(mask_class, gender_class, age_class)

        return total_class