import torch
from torch import nn
from torch.nn import functional as F

class OrthoFeatLoss(nn.Module):
    def __init__(self, pairs):
        super().__init__()
        self.loss_fn = F.cosine_similarity
        self.pairs = pairs
    
    def forward(self, features, l1=False):
        loss = []
        for idx1, idx2 in self.pairs:
            features1, features2 = features[idx1], features[idx2]
            for feature1, feature2 in zip(features1, features2):
                if l1:
                    loss.append(self.loss_fn(feature1, feature2, dim=-1).abs().mean())
                else:
                    loss.append(self.loss_fn(feature1, feature2, dim=-1).square().mean())
        return torch.stack(loss, dim=0).mean()
    
class OrthoParamLoss(nn.Module):
    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs
    
    def forward(self, params):
        loss = 0
        for idx1, idx2 in self.pairs:
            params1, params2 = params[idx1], params[idx2]
            for param1, param2 in zip(params1, params2):
                loss += torch.norm(torch.mm(param1, param2), p=1)
        return loss