
import torch.nn as nn

def create_loss(weight=None, reduction='mean'):
    return nn.BCELoss(weight=weight,reduction=reduction)
