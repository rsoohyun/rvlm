from torch import nn

from .clip import *
from .model import *

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
        
    def set_desc_loss_fn(self, target_layers, desc_emb, loss_fn=None, l1=False):
        self.model.visual.desc_emb = desc_emb
        self.model.visual.loss_fn = loss_fn
        self.model.visual.l1 = l1
        self.model.visual.target_layers = target_layers
    
    def forward(self, x):
        out, desc_loss, feat_loss = self.model.encode_image(x)
        return self.fc(out), desc_loss, feat_loss