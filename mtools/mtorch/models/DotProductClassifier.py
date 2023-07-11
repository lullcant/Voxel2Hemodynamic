import torch.nn as nn
from os import path


class DotProduct_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x, *args):
        x = self.fc(x)
        return x


def create_model(in_features, out_features=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    clf = DotProduct_Classifier(feat_dim=in_features, num_classes=out_features)
    return clf
