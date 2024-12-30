import torch


class Hooker():
    def __init__(self, model, num_lora, desc_emb, loss_fn, l1=False):
        self.num_lora = num_lora
        self.early_exit_proj = model.model.visual.early_exit_proj
        self.desc_emb = desc_emb
        
        self.loss = []
        self.loss_fn = loss_fn
        self.l1 = l1
        
    def clear(self):
        self.loss = []
        
    def compute_loss(self, image_embs):
        image_embs = {k: self.early_exit_proj(v.to("cuda")) for k,v in image_embs.items() if k != 'org'}
        image_embs = {k: v / v.norm(dim=-1, keepdim=True) for k,v in image_embs.items()}
        image_embs = {k: [v @ self.desc_emb.t()] for k,v in image_embs.items()}
        
        self.loss.append(self.loss_fn(image_embs, self.l1))
        
    def return_loss(self):
        return torch.stack(self.loss, dim=0).mean()