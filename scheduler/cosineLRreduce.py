from torch.optim.lr_scheduler import _LRScheduler
import math


class cosineLRreduce(_LRScheduler):

    def __init__(self, optimizer, K):
        self.optimizer = optimizer
        self.K = K
        super(cosineLRreduce, self).__init__(optimizer)

    def get_lr(self):
        if self._step_count - 2 > self.K:
            print("scheduler error, k > K")
        weight = math.cos(7 * math.pi / 16 / self.K * (self._step_count-2))
        return [base_lr*weight for base_lr in self.base_lrs]
