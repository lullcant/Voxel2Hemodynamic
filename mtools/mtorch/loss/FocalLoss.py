import torch.nn.functional as F
import torch.nn as nn
import torch


def focal_loss(input_values, gamma, mean='average'):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    if mean == 'average':
        return loss.mean()
    elif mean == 'sum':
        return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., mean='average'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.mean = mean

    def forward(self, inputs, target, *args):
        return focal_loss(
            F.cross_entropy(
                inputs, target, reduction='none', weight=self.weight
            ), gamma=self.gamma, mean=self.mean
        )


def create_loss(weight=None, gamma=2., mean='average'):
    return FocalLoss(weight=weight, gamma=gamma, mean=mean)
