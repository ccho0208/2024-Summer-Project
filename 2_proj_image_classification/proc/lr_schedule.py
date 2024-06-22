import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from proc.files import check_path


"""
    Custom CosineAnnealingUpRestarts
    - PyTorch code: https://koreapy.tistory.com/787
    - Paper: SGDR, Stochastic Gradient Descent with Warm Restarts (https://arxiv.org/abs/1608.03983)
    - The following class should be added to the following file,
        ~/venv_torch/lib/python3.x/site-packages/torch/optim/lr_scheduler.py
        
    - Args:
        optimizer (Optimizer): Wrapped optimizer
        T_0 (int): Number of iterations for the first restart
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1
        eta_max (float, optional): Maximum learning rate. Default 0.1
        T_up (int, optional): Linear warm-up step size, Default 0
        gamma (float, optional): Decrease rate of max learning rate by cycle. Default: 1
        last_epoch (int, optional): The index of last epoch. Default: -1


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
"""


def plot_lr_curve():
    epochs = 200        # default: 200
    lr = 0.1              # default: 0.1

    # 1) set dummy model & optimizer
    model = torch.nn.Linear(2,1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 2) initiate learn-rate scheduler
    # a) CosineAnnealingLR
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs/5))

    # b) CyclicLR
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,
                                                  step_size_up=int(epochs/10),
                                                  mode='triangular2')

    # c) CosineAnnealingWarmRestarts
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0.001)

    # d) Custom CosineAnnealingWarmUpRestarts
    # `lr` for `optimizer` should be zero
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmUpRestarts(optimizer,
    #                                                             T_0=50,
    #                                                             T_mult=2,
    #                                                             eta_max=0.1,
    #                                                             T_up=1,
    #                                                             gamma=0.5)

    # 3) compute learn-rate schedule
    lrs = []
    for epoch in range(epochs):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # plot the learn-rate schedule
    ind_zeros = [k+1 for k,e in enumerate(lrs) if e==0]
    print('epochs whose learn-rate is zero: {}'.format(ind_zeros))

    HOME = os.path.expanduser('~')
    spath = os.path.join(HOME,'Unzipped/!exp_prac/0_prac/1_cifar10/results')
    check_path(spath)
    fn = os.path.join(spath, 'lr_curve.png')
    plt.figure(); plt.clf()
    plt.plot(lrs, color='blue')
    plt.grid()
    ax = plt.gca()
    ax.set_xlabel('epoch')
    ax.set_ylabel('learn-rate')

    # save the plot to file
    plt.savefig(fn)
    plt.close()


if __name__ == '__main__':
    plot_lr_curve()
