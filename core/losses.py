import torch
from torch import nn
from torch.nn import functional as F

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = F.cosine_similarity
    
    def forward(self, features_list, l1=False):
        loss = []
        num_features = len(features_list)
        for i in range(num_features):
            for j in range(i + 1, num_features):
                feature1, feature2 = features_list[i], features_list[j]
                if l1:
                    loss.append(self.loss_fn(feature1, feature2, dim=-1).abs().mean())
                else:
                    loss.append(self.loss_fn(feature1, feature2, dim=-1).square().mean())
        return torch.stack(loss, dim=0).mean() if loss else torch.tensor(0.0, device=features_list[0].device)

class ParameterOrthogonalityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model):
        orthogonal_loss = 0.0

        for module in model.modules():
            if isinstance(module, LoRAInjectedLinear):
                lora_a_params = [getattr(module, f"lora{i}_A").weight for i in range(module.num_lora)]
                for i in range(len(lora_a_params)):
                    for j in range(i + 1, len(lora_a_params)):
                        param_i = lora_a_params[i]
                        param_j = lora_a_params[j]
                        # Frobenius norm of the dot product
                        dot_product = torch.mm(param_i, param_j.T)
                        orthogonal_loss += torch.norm(dot_product, p="fro")

        return orthogonal_loss

