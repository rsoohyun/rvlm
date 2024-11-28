import torch
from torch import nn
from torch.nn import functional as F

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = F.cosine_similarity
    
    def forward(self, features1, features2, l1=False):
        loss = []
        for feature1, feature2 in zip(features1, features2):
            if l1:
                loss.append(self.loss_fn(feature1, feature2, dim=-1).abs().mean())
            else:
                loss.append(self.loss_fn(feature1, feature2, dim=-1).square().mean())
        return torch.stack(loss, dim=0).mean()