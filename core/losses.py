import torch
from torch import nn
from torch.nn import functional as F
from core import loralib

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = F.cosine_similarity

    def forward(self, all_features, num_lora, l1=False):
        loss = []
        lora_indices = [f"lora{i}" for i in range(num_lora)]  

        # Extract layer paths (e.g., "22.attn.k_proj")
        layers = set([".".join(key.split('.')[:-1]) for key in all_features.keys() if key.endswith('_A')])

        for layer in layers:
            # Collect features for all LoRAs in this layer
            layer_features = {
                lora: all_features[f"{layer}.{lora}_A"]
                for lora in lora_indices
                if f"{layer}.{lora}_A" in all_features
            }

            # Compute pairwise losses between LoRAs
            for i, lora_i in enumerate(lora_indices):
                for j, lora_j in enumerate(lora_indices):
                    if i < j and lora_i in layer_features and lora_j in layer_features:
                        feature1 = layer_features[lora_i]
                        feature2 = layer_features[lora_j]
                        if l1:
                            loss.append(self.loss_fn(feature1, feature2, dim=-1).abs().mean())
                        else:
                            loss.append(self.loss_fn(feature1, feature2, dim=-1).square().mean())

        return torch.stack(loss, dim=0).mean()




class ParameterOrthogonalityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model):
        orthogonal_loss = 0.0

        for module in model.modules():
            if isinstance(module, loralib.LoRAInjectedLinear):
                lora_a_params = [getattr(module, f"lora{i}_A").weight for i in range(module.num_lora)]
                for i in range(len(lora_a_params)):
                    for j in range(i + 1, len(lora_a_params)):
                        param_i = lora_a_params[i]
                        param_j = lora_a_params[j]
                        # Frobenius norm of the dot product
                        dot_product = torch.mm(param_i, param_j.T)
                        orthogonal_loss += torch.norm(dot_product, p="fro")

        return orthogonal_loss

