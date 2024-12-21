import torch
from torch import nn
from torch.nn import functional as F

class OrthoFeatLoss(nn.Module):
    def __init__(self, pairs, kl=False):
        super().__init__()
        if kl: self.loss_fn = F.kl_div
        else: self.loss_fn = F.cosine_similarity
        self.kl = kl
        self.pairs = pairs
    
    def forward(self, features, l1=False):
        loss = []
        for idx1, idx2 in self.pairs:
            features1, features2 = features[idx1], features[idx2]
            for feature1, feature2 in zip(features1, features2):
                if self.kl:
                    loss.append(self.loss_fn(feature1, feature2, reduction="batchmean"))
                elif l1:
                    loss.append(self.loss_fn(feature1, feature2, dim=-1).abs().mean())
                else:
                    loss.append(self.loss_fn(feature1, feature2, dim=-1).square().mean())
        return torch.stack(loss, dim=0).mean()
    
class OrthoParamLoss(nn.Module):
    def __init__(self, pairs, compare_org=False):
        super().__init__()
        self.pairs = pairs
        self.compare_org = compare_org
    
    def forward(self, params, org_params=None):
        loss = 0
        for idx1, idx2 in self.pairs:
            params1, params2 = params[idx1], params[idx2]
            for param1, param2 in zip(params1, params2):
                loss += torch.abs(torch.mm(param1, param2.T)).sum()
        if self.compare_org:
            for lora_params in params.values():
                for lora_param, org_param in zip(lora_params, org_params):
                    loss += torch.abs(torch.mm(org_param, lora_param.T)).sum()
        return loss