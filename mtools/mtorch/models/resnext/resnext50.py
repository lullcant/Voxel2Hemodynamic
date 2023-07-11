from mtool.mtorch.models.resnext.ResNextFeature import ResNext, Bottleneck


def create_model(use_fc=False, dropout=None, last_relu=False, *args):
    print('Loading resnet50. use_fc:{} dropout:{}'.format(use_fc, dropout))
    resnext = ResNext(Bottleneck, [3, 4, 6, 3], use_fc=use_fc, dropout=None,
                      groups=32, width_per_group=4, last_relu=last_relu)
    return resnext


if __name__ == '__main__':
    import torch

    x = torch.rand(size=[5, 3, 224, 224])
    model = create_model()
    print(model(x).size())
