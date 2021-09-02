from torch import functional, torch, nn
import torch.nn.functional as F

# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
# fetched from baseline
class LJH_FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

class cosineloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = torch.nn.CosineEmbeddingLoss(margin=0.25, size_average=None, reduce=None, reduction='mean')

    def forward(self, tensor_A, tensor_B, label_A, label_B):
        labels = (label_A==label_B).int()

        return self.cosine(tensor_A, tensor_B, labels)

cust_FocalLoss = LJH_FocalLoss()

# class labelsmoothloss(nn.Module):
#     def __init__(self, label_smoothing=0., reduction = 'sum'):
#         nn.Module.__init__(self)
#         self.smooth = label_smoothing
#         self.reduction = reduction

#     def forward(self, input_tensor, target_tensor):
#         log_prob = F.log_softmax(input_tensor, dim=-1)
#         prob = torch.exp(log_prob)
#         target_tensor = F.one_hot(target_tensor, num_classes=18)
#         target_tensor = target_tensor*(1-self.smooth) + self.smooth / 18

#         return nn.BCEWithLogitsLoss(prob, target_tensor, reduction=self.reduction)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

cust_labelsmoothloss = LabelSmoothingLoss(18, 0.05)