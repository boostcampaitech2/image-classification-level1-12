from sklearn.metrics import f1_score
import torch
from torch import Tensor


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def f1(output: Tensor, target: Tensor, average="macro"):
    pred = torch.argmax(output, dim=1)
    return f1_score(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), average=average)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
