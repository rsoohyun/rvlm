from torch import nn

from .clip import *

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
        
    def forward(self, x):
        out = self.model.encode_image(x)
        return self.fc(out)