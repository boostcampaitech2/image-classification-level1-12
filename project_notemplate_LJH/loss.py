import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    return F.cross_entropy(output, target)

def F1_loss(output, target):
    return F.l1_loss(output, F.one_hot(target, output.shape[1]))


# import torch
# o = torch.ones((64, 2))
# t = torch.ones((64)).long()

# print(F1_loss(o, t))