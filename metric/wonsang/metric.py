from sklearn.metrics import f1_score
import torch
from torch import Tensor
from ..baseline.metric import accuracy


def ws_age_to_age_class(age: Tensor) -> Tensor:
    age_long = age.long()
    age_class = torch.where(age_long < 29.5,
                            torch.full_like(age_long, 0, device=age_long.device), age_long)
    age_class = torch.where(torch.logical_and(age_long >= 29.5, age_long < 59.5),
                            torch.full_like(age_long, 1, device=age_long.device), age_class)
    age_class = torch.where(age_long >= 59.5,
                            torch.full_like(age_long, 2, device=age_long.device), age_class)
    return age_class


def ws_total_accuracy_for_2_classifier_1_regression(output: Tensor, target: Tensor) -> float:
    """
    :param output
    tuple :3
     Tensor: (64, 3)
     Tensor: (64, 2)
     Tensor: (64,)
    :param target
    Tensor: (64, 3)
    """
    with torch.no_grad():
        mask_pred = torch.argmax(output[:, :3], dim=1)
        mask_correct = torch.eq(mask_pred, target[:, 0])

        gender_pred = torch.argmax(output[:, 3:5], dim=1)
        gender_correct = torch.eq(gender_pred, target[:, 1].long())

        age_pred_class = ws_age_to_age_class(output[:, 5])
        age_target_class = ws_age_to_age_class(target[:, 2])

        age_correct = torch.eq(age_pred_class, age_target_class)

        total_correct = torch.logical_and(torch.logical_and(mask_correct, gender_correct), age_correct)

        num_correct = torch.sum(total_correct).item()

    return num_correct / len(total_correct)


def ws_mask_accuracy_from_3_output(output: Tensor, target: Tensor) -> float:
    return accuracy(output[:, :3], target[:, 0].long())


def ws_gender_accuracy_from_3_output(output: Tensor, target: Tensor) -> float:
    return accuracy(output[:, 3:5], target[:, 1].long())


def ws_age_accuracy_from_3_output(output: Tensor, target: Tensor) -> float:
    # return accuracy(output[2], target[2])
    age_pred_class = ws_age_to_age_class(output[:, 5]).view(-1)
    age_target_class = ws_age_to_age_class(target[:, 2]).view(-1)

    correct = 0
    correct += torch.sum(age_pred_class == age_target_class).item()

    return correct / len(age_target_class)


def ws_f1_from_3_output(output: Tensor, target: Tensor, average="macro") -> float:
    with torch.no_grad():
        mask_pred = torch.argmax(output[:, :3], dim=1)
        gender_pred = torch.argmax(output[:, 3:5], dim=1)
        age_pred = ws_age_to_age_class(output[:, 5])
        total_pred = torch.mul(mask_pred, 6) + torch.mul(gender_pred, 3) + age_pred

        mask_target = target[:, 0].long()
        gender_target = target[:, 1].long()
        age_target = ws_age_to_age_class(target[:, 2])
        total_target = torch.mul(mask_target, 6) + torch.mul(gender_target, 3) + age_target

    return f1_score(total_pred.detach().cpu().numpy(), total_target.detach().cpu().numpy(), average=average)
