# import torch
# from Models.wideresnet import WideResNet
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import math


class cosineLRreduce(_LRScheduler):

    def __init__(self, optimizer, K):
        self.optimizer = optimizer
        self.K = K
        super(cosineLRreduce, self).__init__(optimizer)

    def get_lr(self):
        if self._step_count - 1 > self.K:
            print("scheduler error, k > K")
        weight = math.cos(7 * math.pi / 16 / self.K * (self._step_count-1))
        return [base_lr*weight for base_lr in self.base_lrs]

def train_model(model, trainloader, criterion, augmentationclass):
    pass



if __name__ == "__main__":
    from Models.wideresnet import WideResNet
    import torch
    model = WideResNet(3, 28, 2, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = cosineLRreduce(optimizer, 20)
    for i in range(20):
        print(scheduler._step_count)
        print(scheduler.get_lr())
        scheduler.step()

    