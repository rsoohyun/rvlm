import torch
from torch import nn
from torch.nn import functional as F


def js_divergence(x, y, tau=0.01, eps=1e-8):
    p = F.softmax(x / tau, dim=-1)
    q = F.softmax(y / tau, dim=-1)
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(m.log(), p, reduction='batchmean') \
         + 0.5 * F.kl_div(m.log(), q, reduction='batchmean')

def compute_entropy(x, tau=0.01, eps=1e-8):
    p = F.softmax(x / tau, dim=-1)
    return -(p * p.log()).sum(dim=-1)

def pairwise_dot(x, y, tau=0.01, eps=1e-8):
    p = F.softmax(x / tau, dim=-1)
    q = F.softmax(y / tau, dim=-1)
    return (p * q).sum(dim=-1)


class OrthoFeatLoss(nn.Module):
    def __init__(self, pairs, args):
        super().__init__()
        if args.kl: self.loss_fn = F.kl_div
        else: self.loss_fn = F.cosine_similarity
        self.kl = args.kl
        self.entropy = args.entropy
        self.dot = args.dot
        self.pairs = pairs
        self.tau = 0.1
        self.eps = 1e-8

    def forward(self, features, l1=False):
        loss = []
        for idx1, idx2 in self.pairs:
            features1, features2 = features[idx1], features[idx2]
            if self.dot:
                dot_val = pairwise_dot(features1, features2, tau=self.tau).mean()
                if self.entropy:
                    dot_val += compute_entropy(features1, tau=self.tau).mean() \
                            + compute_entropy(features2, tau=self.tau).mean()
                loss.append(dot_val)
            elif self.kl:
                loss.append(js_divergence(features1, features2, self.tau))
                #loss.append((-1) * self.loss_fn(F.log_softmax(features1/self.tau, dim=-1), F.softmax(features2/self.tau, dim=-1), reduction="batchmean"))
            elif l1:
                loss.append(self.loss_fn(features1, features2, dim=-1).abs().mean())
            else:
                loss.append(self.loss_fn(features1, features2, dim=-1).square().mean())
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