import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from itertools import combinations
from collections import defaultdict

import wandb

from data import load_dataset, imagenet_templates
from core import clip, loralib, losses, utils
from eval import evaluate, infer
from utils import *



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP")
    parser.add_argument("--dataset", type=str, default="waterbird", choices=['waterbird', 'celeba'])
    parser.add_argument("--n_cls", type=int, default=2)
    parser.add_argument("--prompt_id", type=int, default=0)
    
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--num_lora", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--lora_modules", type=str, default="q,v")
    parser.add_argument("--lora_w_pretrain", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--dot", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=5e-5)
    
    parser.add_argument("--lambda_cls", type=float, default=1.)
    parser.add_argument("--lambda_ortho", type=float, default=1.)
    parser.add_argument("--l1", action="store_true")
    parser.add_argument("--last_num", type=int, default=24)
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@LoRA_desc")
    
    parser.add_argument("--resume_id", type=str, default="")
    args = parser.parse_args()
    
    ## Set ENV
    utils.set_seed(args.seed)
    
    lora_idxs = list(range(args.num_lora))
    lora_pairs = list(combinations(lora_idxs, 2))
    lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]
    
    save_dir = args.save_dir
    save_dir += f"@{'_'.join(lora_modules)}"
    if args.lora_w_pretrain: save_dir += "@wp"
    save_dir += f"@r{args.r}/"
    os.makedirs(save_dir, exist_ok=True)
    write_json(f"{save_dir}config.json", vars(args))
    
    if args.resume_id:
        wandb.init(project="rvlm", id=args.resume_id, resume=True)
    else:
        wandb.init(project="rvlm")
    wandb.config.update(args)
    wandb.run.name = save_dir.split('/')[-2]
    
    ## Load data and model
    if args.arch == "CLIP":
        model = clip.CLIP_FT("ViT-L/14", "cuda", n_cls=args.n_cls)
    else:
        raise NotImplementedError(f'{args.arch} is not implemented yet.')
    print('{} w/o LoRA: {:.1f}M'.format(args.arch, sum(param.numel() for param in model.parameters())/1000000.0))
    
    loralib.apply_lora(model, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules=lora_modules)
    print('{} w/  LoRA: {:.1f}M'.format(args.arch, sum(param.numel() for param in model.parameters())/1000000.0))
    
    train_dataset = load_dataset(args.data_dir, args.dataset, "train", model.preprocess, args.prompt_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    test_dataset = load_dataset(args.data_dir, args.dataset, "test", model.preprocess, args.prompt_id)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=args.num_workers)

    cls_loss_fn = nn.CrossEntropyLoss()
    ortho_loss_fn = losses.OrthoFeatLoss(lora_pairs, args=args)
    
    ## Train
    wandb.define_metric("step/iter")
    wandb.define_metric("step/*", step_metric="step/iter")
    loralib.set_used_lora(model, lora_idxs)
    _, trainable_params = loralib.get_lora_params(model, fc=True, idxs=lora_idxs)
    
    if args.optim=="adamw": optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    elif args.optim=="sgd": optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    all_descs = [[template.format(f"bird with {desc.lower()}") for template in imagenet_templates] for desc in train_dataset.all_descs]
    all_descs = [model.model.encode_text(clip.tokenize(desc).to("cuda")) for desc in all_descs]
    all_descs = [desc / desc.norm(dim=-1, keepdim=True) for desc in all_descs]
    all_descs = [desc.mean(dim=0) for desc in all_descs]
    all_descs = [desc / desc.norm() for desc in all_descs]
    desc_feats = torch.stack(all_descs, dim=0).detach()
    
    target_layers = [str(i) for i in range(24-args.last_num, 24)]
    model.set_hooker(target_layers, args.num_lora, desc_feats, ortho_loss_fn, args.l1)
    
    iteration = 0
    train_losses, train_cls_losses, train_ortho_losses = [], [], []
    train_preds, train_labels, train_spurious = [], [], []
    
    for epoch in range(1, args.epochs+1):
        if os.path.exists(save_dir + f'epoch{epoch}.pt'):
            loralib.load_lora(model, save_dir + f'epoch{epoch}.pt')
            optimizer.load_state_dict(torch.load(save_dir + f'epoch{epoch}_op.pt'))
            iteration = epoch * len(train_loader)
            continue
        
        for data in tqdm(train_loader, f'Epoch: {epoch:03d}'):
            images, attrs, _ = data
            
            model.hooker.clear()
            outputs = model(images.to("cuda"))
            
            cls_loss = cls_loss_fn(outputs, attrs[:,0].to("cuda"))
            ortho_loss = model.hooker.return_loss()
            loss = args.lambda_cls * cls_loss + args.lambda_ortho * ortho_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_cls_losses.append(cls_loss.item())
            train_ortho_losses.append(ortho_loss.item())
            _, preds = torch.max(outputs, 1)
            train_preds.append(preds)
            train_labels.append(attrs[:,0])
            train_spurious.append(attrs[:,1])

            iteration += 1

            if iteration % 10 == 0:
                lr = [group['lr'] for group in optimizer.param_groups][0]
                train_loss = np.mean(train_losses)
                train_cls_loss = np.mean(train_cls_losses)
                train_ortho_loss = np.mean(train_ortho_losses)
                
                train_preds, train_labels, train_spurious = torch.concat(train_preds, dim=0).detach().cpu().numpy(), torch.concat(train_labels, dim=0).detach().cpu().numpy(), torch.concat(train_spurious, dim=0).detach().cpu().numpy()
                train_worst_acc, train_avg_acc, _ = evaluate(train_preds, train_labels, train_spurious)
                
                wandb.log({
                    "step/iter": iteration,
                    "step/loss": train_loss,
                    "step/loss_cls": train_cls_loss,
                    "step/loss_ortho": train_ortho_loss,
                    "step/train_worst_acc": train_worst_acc,
                    "step/train_avg_acc": train_avg_acc,
                })
                # print(f'\nIteration: {iteration:06d}, LR: {lr:.06f}, L: {train_loss:.03f}, L_cls: {train_cls_loss:.03f}, L_ortho: {train_ortho_loss:.03f}')

                train_losses, train_cls_losses, train_ortho_losses = [], [], []
                train_preds, train_labels, train_spurious = [], [], []

        loralib.save_lora(model, save_dir + f'epoch{epoch}.pt', idxs=lora_idxs)
        torch.save(optimizer.state_dict(), save_dir + f'epoch{epoch}_op.pt')
    
        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, desc="Eval CLIP+LoRA"))
        print(f"Epoch {epoch}) Test Set - Average Accuracy: {avg_acc:.2f}, Worst Group Accuracy: {worst_acc:.2f}")
        print(f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]")
        model.train()
    wandb.finish()