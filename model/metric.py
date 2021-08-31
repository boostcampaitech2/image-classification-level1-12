from typing import Tuple

from sklearn.metrics import f1_score
import torch
from torch import Tensor



def age_to_age_class(age: Tensor):
    age_round = age.round()
    age_class = torch.where(age_round < 29.5, torch.full_like(age_round, 0, device=age_round.device), age_round)
    age_class = torch.where(torch.logical_and(age_round >= 29.5, age_round < 59.5),
                            torch.full_like(age_round, 1, device=age_round.device), age_class)
    age_class = torch.where(age_round >= 59.5, torch.full_like(age_round, 2, device=age_round.device), age_class)
    return age_class


def mask_total_accuracy(output: Tuple[Tensor, Tensor, Tensor], target: Tuple[Tensor, Tensor, Tensor]):
    with torch.no_grad():
        mask_pred = torch.argmax(output[0], dim=1)
        mask_correct = torch.eq(mask_pred, target[0])

        gender_pred = torch.argmax(output[1], dim=1)
        gender_correct = torch.eq(gender_pred, target[1])

        age_pred_class = age_to_age_class(output[2]).view(-1)
        age_target_class = age_to_age_class(target[2]).view(-1)

        age_correct = torch.eq(age_pred_class, age_target_class)

        total_correct = torch.logical_and(torch.logical_and(mask_correct, gender_correct), age_correct)

        num_correct = torch.sum(total_correct)

    return num_correct / len(total_correct)


def mask_accuracy(output: Tuple[Tensor, Tensor, Tensor], target: Tuple[Tensor, Tensor, Tensor]):
    return accuracy(output[0], target[0])


def gender_accuracy(output: Tuple[Tensor, Tensor, Tensor], target: Tuple[Tensor, Tensor, Tensor]):
    return accuracy(output[1], target[1])


def age_accuracy(output: Tuple[Tensor, Tensor, Tensor], target: Tuple[Tensor, Tensor, Tensor]):
    # return accuracy(output[2], target[2])
    age_pred_class = age_to_age_class(output[2]).view(-1)
    age_target_class = age_to_age_class(target[2]).view(-1)

    correct = 0
    correct += torch.sum(age_pred_class == age_target_class).item()

    return correct / len(age_target_class)


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def f1(output: Tuple[Tensor, Tensor, Tensor], target: Tuple[Tensor, Tensor, Tensor], average="macro"):
    with torch.no_grad():
        mask_pred = torch.argmax(output[0], dim=1)
        gender_pred = torch.argmax(output[1], dim=1)
        age_pred = age_to_age_class(output[2]).view(-1)
        total_pred = mask_pred * 6 + gender_pred * 3 + age_pred

        total_target = target[0] * 6 + target[1] * 3 + age_to_age_class(target[2]).view(-1)

    return f1_score(total_pred.detach().cpu().numpy(), total_target.detach().cpu().numpy(), average=average)
