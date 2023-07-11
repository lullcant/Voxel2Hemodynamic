from mtools.mtorch.models.resnet.ResNetFeature import ResNet, BasicBlock


def create_model(use_fc=False, dropout=None, *args):
    print('Loading resnet10. use_fc:{} dropout:{}'.format(use_fc, dropout))
    model = ResNet(BasicBlock, [1, 1, 1, 1], use_fc=use_fc, dropout=dropout)
    return model
