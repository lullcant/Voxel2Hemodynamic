## https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4
import math


class CustomizeScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, T_max, eta_min):
        self.optimizer = optimizer
        self.n_steps = 0
        self.lr = self.optimizer.param_groups[0]['lr']
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1

        # lr = self.eta_min + 0.5 * (self.lr - self.eta_min) * (1 + math.cos(self.n_steps / self.T_max * math.pi))
        if self.n_steps < self.T_max:
            lr = self.eta_min + 0.5 * (self.lr - self.eta_min) * (1 + math.cos(self.n_steps / self.T_max * math.pi))
        else:
            lr = self.eta_min

        for pg in self.optimizer.param_groups:
            if 'lr_scale' in pg.keys():
                pg['lr'] = lr * pg['lr_scale']
            else:
                pg['lr'] = lr

    def state_dict(self):
        return {'step': self.n_steps, 'lr': self.lr}


if __name__ == '__main__':
    import torch
    from mtools.mtorch.models.Unet import create_model

    optim = torch.optim.Adam(create_model().parameters(), lr=0.5)
    sched = CustomizeScheduledOptim(optim, T_max=90, eta_min=0.01)

    x, y = [], []
    for i in range(180):
        sched.step()
        x.append(i)
        y.append(optim.param_groups[0]['lr'])

    import matplotlib.pyplot as plt

    print(y[80])

    plt.plot(x, y)
    plt.show()
