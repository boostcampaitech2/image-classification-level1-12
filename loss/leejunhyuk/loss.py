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