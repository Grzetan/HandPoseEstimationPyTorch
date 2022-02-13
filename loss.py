import torch
import torch.nn as nn

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.criterion = nn.MSELoss(reduction='sum')

    def __call__(self, pred, target):
        loss = self.criterion(pred, target)
        return loss / float(pred.shape[0])