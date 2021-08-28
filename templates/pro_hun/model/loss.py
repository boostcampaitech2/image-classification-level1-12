import torch


# 한 Batch에서의 loss 값
def batch_loss(loss, images):
    return loss.item() * images.size(0)
