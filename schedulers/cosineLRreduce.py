from torch.optim.lr_scheduler import _LRScheduler
import math


class cosineLRreduce(_LRScheduler):

    def __init__(self, optimizer, K, warmup=None, from_step=0):
        self.optimizer = optimizer
        self.K = K
        self.start_from = from_step
        self.warm = 1.0*K/100 if warmup is None else warmup*K # default 1/100 of total training
        self.state = 0 # indicates state, 0 init, 1 warmup, 2 cosine
        super(cosineLRreduce, self).__init__(optimizer)

    def get_lr(self):
        c_step = max(1, self._step_count + self.start_from)
        if c_step > self.K:
            print("scheduler error, k > K")
            return [base_lr/(2**10) for base_lr in self.base_lrs] # if error return very small learning rate
        else:
            if c_step < self.warm:
                weight = c_step / self.warm
                self.state = 1
            else:
                weight = 1
                self.state = 2
            weight *= math.cos(7.0 * math.pi / 16 / self.K * (c_step))
        return [base_lr*weight for base_lr in self.base_lrs]
