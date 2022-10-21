import numpy as np
import torch


def cross_entropy_error(y, t):
    delta = 1e-7
    return -torch.sum(t * torch.log(y + delta), dim=1)


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


class MSE(torch.nn.Module):
    def __init__(self, ignore=-1):
        super(MSE, self).__init__()
        self.loss = None
        self.ignore = ignore

    def forward(self, output, label):
        ignore_mask = label != -1

        pred = torch.where(ignore_mask, output, torch.zeros_like(output))
        target = torch.where(ignore_mask, label, torch.zeros_like(label))

        self.loss = torch.square(pred - target)
        self.loss = self.loss.mean(dim=0)
        self.loss = self.loss.mean()

        return self.loss


def focal_error(y, t, alpha=0.25, gamma=2):
    delta = 1e-7
    weight = torch.pow(1.0 - y, gamma)
    focal = -alpha * weight * (t * torch.log(y + delta))
    return torch.sum(focal, dim=1)


class FocalLoss(torch.nn.Module):
    def __init__(self, ignore=-1):
        super(FocalLoss, self).__init__()
        self.loss = None
        self.ignore = ignore
        self.softmax = torch.nn.Softmax2d()

    def forward(self, output, label):
        ignore_mask = label != -1
        pred = torch.where(ignore_mask, output, torch.zeros_like(output))
        target = torch.where(ignore_mask, label, torch.zeros_like(label))
        y = self.softmax(pred)
        self.loss = focal_error(y, target)

        return self.loss
