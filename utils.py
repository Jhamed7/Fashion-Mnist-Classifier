import torch

def calc_accuracy(preds, labels):
    _, pred_max = torch.max(preds, 1)
    return torch.sum(pred_max == labels.data, dtype=torch.float64) / len(preds)