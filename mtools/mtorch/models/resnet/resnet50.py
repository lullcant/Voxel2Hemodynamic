from mtools.mtorch.models.resnet.ResNetFeature import ResNet, Bottleneck


def create_model(use_fc=False, dropout=None, *args):
    print('Loading resnet50. use_fc:{} dropout:{}'.format(use_fc, dropout))
    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], use_fc=use_fc, dropout=dropout)
    return resnet50
