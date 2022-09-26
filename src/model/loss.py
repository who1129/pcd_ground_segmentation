import numpy as np
import torch


def cross_entropy_error(y, t):
    delta = 1e-7
    return -torch.sum(t * torch.log(y + delta)) / y.shape[0]


class SoftmaxWithloss(torch.nn.Module):
    def __init__(self, ignore=-1):
        super(SoftmaxWithloss, self).__init__()
        self.loss = None
        self.ignore = ignore
        self.softmax = torch.nn.Softmax2d()

    def forward(self, output, label):
        ignore_mask = label != -1

        pred = torch.where(ignore_mask, output, torch.zeros_like(output))
        target = torch.where(ignore_mask, label, torch.zeros_like(label))

        y = self.softmax(pred)
        self.loss = cross_entropy_error(y, target)

        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
