from torch.optim.lr_scheduler import _LRScheduler
import math


class cosineLRreduce(_LRScheduler):

    def __init__(self, optimizer, K, warmup=None):
        self.optimizer = optimizer
        self.K = K
        self.warm = 1.0*K/100 if warmup is None else warmup # default 1/100 of total training
        self.state = 0 # indicates state, 0 init, 1 warmup, 2 cosine
        super(cosineLRreduce, self).__init__(optimizer)

    def get_lr(self):
        c_step = self._step_count -2
        if c_step > self.K:
            print("scheduler error, k > K")
            return [base_lr/(2**10) for base_lr in self.base_lrs] # if error return very small learning rate
        elif c_step < self.warm:
            weight = c_step / self.warm
            self.state = 1
        else:
            weight = math.cos(7.0 * math.pi / 16 / self.K * (c_step - self.warm))
            self.state = 2
        return [base_lr*weight for base_lr in self.base_lrs]
