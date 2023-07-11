import torch
import torch.nn as nn
from mtools.mtorch.loss.SoftDiceLoss import SoftDiceLoss


class ImageLoss(nn.Module):
    def __init__(self, weight, reduction):
        super(ImageLoss, self).__init__()
        self.ce = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
        self.dc = SoftDiceLoss(apply_nonlin=None)

    def forward(self, inputs, target, *args):
        recon, _, _ = inputs
        label, _ = target

        ce_loss = self.ce(recon, label)
        dc_loss = self.dc(recon, label)

        return ce_loss + dc_loss


def create_loss(weight=None, reduction='mean'):
    return ImageLoss(weight, reduction)
