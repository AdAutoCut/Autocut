import torch
import torch.nn.functional as F


def L1_loss(x, y):
    return torch.mean(torch.abs(x - y))


def L2_loss(x, y):
    # refer to https://github.com/EdoardoBotta/RQ-VAE-Recommender/blob/main/modules/loss.py
    # same as torch.nn.MSELoss(reduction='mean')
    return torch.mean((x - y) ** 2)


def cos_loss(x, y):
    cos_sim = F.cosine_similarity(x, y, dim=1)
    loss = (1 - cos_sim).mean()
    return loss


def mixed_loss(x, y):
    loss = L2_loss(x, y)+0.1 * cos_loss(x, y)
    return loss
