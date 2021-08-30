import torch

from data_process.class_converter import mask_to_mask_class, gender_to_gender_class, age_to_age_class, convert_3class_to_1class


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class OneClassTarget:
    def __call__(self, meta_data: dict) -> int:
        mask_class = mask_to_mask_class(meta_data["mask_file_name"])
        gender_class = gender_to_gender_class(meta_data["gender"])
        age_class = age_to_age_class(meta_data["age"])

        total_class = convert_3class_to_1class(mask_class, gender_class, age_class)

        return total_class