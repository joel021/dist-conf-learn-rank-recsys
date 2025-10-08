import torch
import torch.nn as nn

def weighted_mse_with_weight_penalty(y_true, y_pred, reduction='sum'):
    y_true = y_true[:, 0].float()
    y_pred, sample_weight = y_pred[:, 0].float(), y_pred[:, 1].float()

    pred_loss = sample_weight * (y_true - y_pred) ** 2
    det_loss = torch.log(1 / torch.clamp(sample_weight, min=1e-6, max=1e6))
    sample_loss = pred_loss + det_loss

    if reduction == 'none':
        return sample_loss
    elif reduction == 'mean':
        return torch.mean(sample_loss)
    else:  # 'sum'
        return torch.sum(sample_loss)

class WeightedMSEWithWeightPenalty(nn.Module):
    def __init__(self, reduction='sum'):
        super(WeightedMSEWithWeightPenalty, self).__init__()
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        return weighted_mse_with_weight_penalty(y_true, y_pred, reduction=self.reduction)

def custom_mse(y_true, y_pred, reduction='sum'):
    y_true = y_true.float()
    if len(y_pred.shape) > len(y_true.shape):
        y_pred = y_pred[:, 0]
    sample_loss = (y_true - y_pred) ** 2

    if reduction == 'none':
        return sample_loss
    elif reduction == 'mean':
        return torch.mean(sample_loss)
    else:  # 'sum'
        return torch.sum(sample_loss)

class CustomRMSE(nn.Module):
    def __init__(self, reduction='sum'):
        super(CustomRMSE, self).__init__()
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        return torch.sqrt(custom_mse(y_true, y_pred, reduction=self.reduction))

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, outputs, targets):
        return torch.sqrt(nn.MSELoss()(outputs, targets))
