#macro f1
import torch
from sklearn.metrics import f1_score

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def F_score(output, target):
    return f1_score(output.cpu().numpy(), target.cpu().numpy(), average="macro")