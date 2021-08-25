from typing import Tuple

import torch
from torch import FloatTensor, LongTensor, Tensor


def mask_total_accuracy(output: Tuple[Tensor, Tensor, Tensor], target: Tuple[Tensor, Tensor, Tensor]):
    with torch.no_grad():
        mask_pred = torch.argmax(output[0], dim=1)
        gender_pred = torch.argmax(output[1], dim=1)
        age_pred = torch.argmax(output[2], dim=1)

        mask_correct = torch.eq(mask_pred, target[0])
        gender_correct = torch.eq(gender_pred, target[1])
        age_correct = torch.eq(age_pred, target[2])

        total_correct = torch.bitwise_and(torch.bitwise_and(mask_correct, gender_correct), age_correct)

        num_correct = torch.sum(total_correct)

    return num_correct / len(total_correct)


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
