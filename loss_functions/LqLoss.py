#https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py
import torch.nn.functional as F
from torch import abs as t_abs

class LqLoss:

    def __init__(self, q=0.7, num_classes=10):
        self.q = q
        self.num_classes = num_classes

    def __call__(self, logits, targets):
        p = logits.softmax(1)
        t = F.one_hot(targets, num_classes=self.num_classes)
        return t_abs(t - p.pow(self.q)).sum()/self.q

