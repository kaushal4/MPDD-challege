# Utils/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=0.5, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        preds:softmax output
        labels:true values
        """
        ce_loss = F.cross_entropy(preds, targets, reduction='mean')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")