#https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LqLoss(nn.Module):

    def __init__(self, q=0.7):
        super(LqLoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q).sum()
        
        return loss

