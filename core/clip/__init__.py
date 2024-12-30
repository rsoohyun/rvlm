from torch import nn

from .clip import *
from .model import *
from .loss_hooker import Hooker

class CLIP_FT(nn.Module):
    def __init__(self, model_arch, device, n_cls=2):
        super().__init__()
        
        self.model, self.preprocess = load(model_arch, device=device)
        self.model.float()      # mixed precision -> underflow/overflow in optimizer.step()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Linear(self.model.visual.output_dim, n_cls)
        self.fc.to(device).to(self.model.dtype)
        for param in self.fc.parameters():
            param.requires_grad = True
        
        self.device = device
        
    def set_hooker(self, target_layers, num_lora, desc_emb, loss_fn=None, l1=False):
        self.hooker = Hooker(self, num_lora, desc_emb, loss_fn, l1)
        
        for name, submodule in self.model.visual.transformer.resblocks.named_modules():
            if isinstance(submodule, ResidualAttentionBlock) and name in target_layers:
                eval(f"self.model.visual.transformer.resblocks[{name}]").hooker = self.hooker
        
    def forward(self, x):
        out = self.model.encode_image(x)
        return self.fc(out)