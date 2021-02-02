# +
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    @property
    def __name__(self):
        return "CELoss"
    def forward(self, y_pred, y_gt):
         return nn.CrossEntropyLoss()(y_pred, y_gt) 
