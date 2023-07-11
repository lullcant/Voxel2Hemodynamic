from mtools.mtorch.mtrainer import TrainerBase


def train():
    '''
    注意：
    1. 在Windows测试时,num_worker不可用，所以需要设置为0,否则dataloader过不去
    '''
    from mtools.mio import get_yaml
    config = get_yaml('./train.yaml')
    trainer = TrainerBase(config=config)
    trainer.run(DEBUG=True)


if __name__ == '__main__':
    train()
