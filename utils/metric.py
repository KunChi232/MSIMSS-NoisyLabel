import torch.nn as nn

class fscore(nn.Module):
    def __init__(self):
        super().__init__()
    @property
    def __name__(self):
        return 'fscore'
    def forward(self, y_pred, y_gt):
        y_pred = torch.softmax(y_pred, dim = 1)
        y_pred = torch.argmax(y_pred, dim = 1)
        
        tp = (y_gt * y_pred).sum().to(torch.float32)
        tn = ((1 - y_gt) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_gt) * y_pred).sum().to(torch.float32)
        fn = (y_gt * (1 - y_pred)).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision*recall) / (precision + recall + epsilon)
        
        return f1