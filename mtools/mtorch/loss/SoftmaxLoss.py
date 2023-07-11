import torch.nn as nn


class SoftmaxLoss(nn.Module):
    def __init__(self, weight, reduction):
        super(SoftmaxLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, inputs, target, *args):
        return self.ce(inputs, target)


def create_loss(weight=None, reduction='mean'):
    return SoftmaxLoss(weight=weight, reduction=reduction)
