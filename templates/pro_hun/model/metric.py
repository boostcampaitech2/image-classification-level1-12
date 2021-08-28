import torch
from sklearn.metrics import f1_score


# 한 Batch에서의 Accuracy 값
def batch_acc(pred, label):
    return torch.sum(pred == label)


# 한 Batch에서의 loss 값
def batch_loss(loss, images):
    return loss.item()*images.size(0)


# 한 Batch에서의 f1_score 값
def batch_f1(pred, label, method):
    return f1_score(pred, label, average=method)


# Epoch당 평균값 반환
def epoch_mean(val, len):
    return val/len