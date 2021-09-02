from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TwoCEOneMSELoss(nn.Module):
    def __init__(
            self,
            mask_class_weight: Tensor = None,
            gender_class_weight: Tensor = None,
            age_weight: float = 0.015):
        super().__init__()
        self.age_weight = age_weight

        self.mask_loss_func = nn.CrossEntropyLoss(weight=mask_class_weight)
        self.gender_loss_func = nn.CrossEntropyLoss(weight=gender_class_weight)
        self.age_loss_func = nn.MSELoss()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        mask_loss = self.mask_loss_func(x[:, :3], target[:, 0].round().long())
        gender_loss = self.gender_loss_func(x[:, 3:5], target[:, 1].round().long())
        age_loss = self.age_loss_func(x[:, 5], target[:, 2])
        total_loss = mask_loss + gender_loss + torch.mul(age_loss, self.age_weight)
        return total_loss


ws_two_ce_one_mse_loss_0015 = TwoCEOneMSELoss(age_weight=0.015)


class AgeMSELoss(nn.Module):
    def __init__(self, lt60_weight: float = 1., gte60_weight: float = 6.):
        super().__init__()
        self.lt60_weight = lt60_weight
        self.gte60_weight = gte60_weight

        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, output: Tensor, target: Tensor):
        weight = torch.heaviside(target.sub(59.5), torch.tensor([0.]).to(target.device))\
            .mul(self.gte60_weight - self.lt60_weight).add(self.lt60_weight)
        mse_loss_val = self.mse_loss(output, target)
        weighted_mse_loss_val = mse_loss_val.mul(weight)
        mean_weighted_mse_loss_val = torch.mean(weighted_mse_loss_val, dim=0)
        return mean_weighted_mse_loss_val


class TwoCEOneWeightedMSELoss(TwoCEOneMSELoss):
    def __init__(
            self,
            mask_class_weight: Tensor = None,
            gender_class_weight: Tensor = None,
            age_weight: float = 0.04,
            lt60_weight: float = 1.,
            gte60_weight: float = 6.
    ):
        super().__init__(mask_class_weight, gender_class_weight, age_weight)

        # change age_loss_func
        self.age_loss_func = AgeMSELoss(lt60_weight=lt60_weight, gte60_weight=gte60_weight)


ws_two_ce_one_weighted_mse_loss_0015 = TwoCEOneWeightedMSELoss(lt60_weight=1., gte60_weight=6., age_weight=0.04)
