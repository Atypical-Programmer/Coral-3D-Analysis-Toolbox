import torch
import torch.nn as nn
from loss.iou import SoftIoULoss

class SegmentationLosses(object):
    def __init__(self, weight=None,  batch_average=True, cuda=False):
        #ignore_index=255:代表着标签为255的像素不会参与训练，也不会参与评价指标的计算
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'iou':
            return self.IoULoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5): #0.5
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def IoULoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = SoftIoULoss(n_classes=3)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss




