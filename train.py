import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import wandb

from data import load_dataset
from core import clip, loralib, losses, utils
from eval import evaluate



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP")
    parser.add_argument("--dataset", type=str, default="waterbird")
    
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--lora_mlp", action="store_true")
    
    parser.add_argument("--epochs_step1", type=int, default=4)
    parser.add_argument("--epochs_step2", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=5e-5)
    
    parser.add_argument("--lambda_cls", type=float, default=1.)
    parser.add_argument("--lambda_ortho", type=float, default=1.)
    parser.add_argument("--l1", action="store_true")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@SepLoRA")
    
    parser.add_argument("--resume_id", type=str, default="")
    args = parser.parse_args()
    
    ## Set ENV
    if args.resume_id:
        wandb.init(project="rvlm", id=args.resume_id, resume=True)
    else:
        wandb.init(project="rvlm")
    wandb.config.update(args)
    wandb.run.name = f"{args.save_dir.split('/')[-1]}@r{args.r}"
    
    utils.set_seed(args.seed)
    save_dir = f"{args.save_dir}@mlp@r{args.r}/" if args.lora_mlp else f"{args.save_dir}@r{args.r}/"
    os.makedirs(save_dir, exist_ok=True)
    
    ## Load data and model
    train_dataset = load_dataset(args.data_dir, args.dataset, "train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    # valid_dataset = load_dataset(args.data_dir, args.dataset, "valid")
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    
    if args.arch == "CLIP":
        model = clip.CLIP_FT("ViT-L/14", "cuda", n_cls=train_dataset.n_classes)
    else:
        raise NotImplementedError(f'{args.arch} is not implemented yet.')
    print('{} w/o LoRA: {:.1f}M'.format(args.arch, sum(param.numel() for param in model.parameters())/1000000.0))
    
    loralib.apply_lora(model, 2, args.r, args.lora_alpha, args.lora_dropout, mlp=args.lora_mlp)
    print('{} w/  LoRA: {:.1f}M'.format(args.arch, sum(param.numel() for param in model.parameters())/1000000.0))
    
    cls_loss_fn = nn.CrossEntropyLoss()
    ortho_loss_fn = losses.OrthogonalLoss()
    
    ## Train
    # Step1
    wandb.define_metric("step1/iter")
    wandb.define_metric("step1/*", step_metric="step1/iter")
    loralib.set_used_lora(model, [0])
    _, trainable_params = loralib.get_lora_params(model, fc=True, idxs=[0])
    optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    
    iteration = 0
    train_losses, train_cls_losses = [], []
    train_preds, train_labels, train_spurious = [], [], []
    for epoch in range(1, args.epochs_step1+1):
        if os.path.exists(save_dir + f'step1_epoch{epoch}.pt'):
            loralib.load_lora(model, save_dir + f'step1_epoch{epoch}.pt')
            optimizer.load_state_dict(torch.load(save_dir + f'step1_epoch{epoch}_op.pt'))
            iteration = epoch * len(train_loader)
            continue
        
        for data in tqdm(train_loader, f'Step1 Epoch: {epoch:03d}'):
            images, attrs, _ = data
            
            outputs = model(images.to("cuda"))
            cls_loss = cls_loss_fn(outputs, attrs[:,0].to("cuda"))
            
            loss = cls_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_cls_losses.append(cls_loss.item())
            _, preds = torch.max(outputs, 1)
            train_preds.append(preds)
            train_labels.append(attrs[:,0])
            train_spurious.append(attrs[:,1])

            iteration += 1

            if iteration % 10 == 0:
                lr = [group['lr'] for group in optimizer.param_groups][0]
                train_loss = np.mean(train_losses)
                train_cls_loss = np.mean(train_cls_losses)
                
                train_preds, train_labels, train_spurious = torch.concat(train_preds, dim=0).detach().cpu().numpy(), torch.concat(train_labels, dim=0).detach().cpu().numpy(), torch.concat(train_spurious, dim=0).detach().cpu().numpy()
                train_worst_acc, train_avg_acc, _ = evaluate(train_preds, train_labels, train_spurious)
                
                wandb.log({
                    "step1/iter": iteration,
                    "step1/loss": train_loss,
                    "step1/loss_cls": train_cls_loss,
                    "step1/train_worst_acc": train_worst_acc,
                    "step1/train_avg_acc": train_avg_acc,
                })
                print(f'\nIteration: {iteration:06d}, LR: {lr:.06f}, L: {train_loss:.03f}, L_cls: {train_cls_loss:.03f}')

                train_losses, train_cls_losses = [], []
                train_preds, train_labels, train_spurious = [], [], []

        loralib.save_lora(model, save_dir + f'step1_epoch{epoch}.pt', idxs=[0])
        torch.save(optimizer.state_dict(), save_dir + f'step1_epoch{epoch}_op.pt')
            
    
    # Step2
    wandb.define_metric("step2/iter")
    wandb.define_metric("step2/*", step_metric="step2/iter")
    loralib.set_used_lora(model, [1])
    _, trainable_params = loralib.get_lora_params(model, fc=True, idxs=[1])
    optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    
    all_features = {}
    def get_output(name):
        def hook(model, input, output):
            all_features[name] = output
        return hook
    
    for name, submodule in model.model.visual.transformer.resblocks.named_modules():
        idx = name.split('.')[0]
        param = '.'.join(name.split('.')[1:])
        if ("lora" in name) and name.endswith('_A'): 
            eval(f"model.model.visual.transformer.resblocks[{idx}].{param}").register_forward_hook(get_output(name))
    
    iteration = 0
    train_losses, train_cls_losses, train_ortho_losses = [], [], []
    train_preds, train_labels, train_spurious = [], [], []
        
    for epoch in range(1, args.epochs_step2+1):
        if os.path.exists(save_dir + f'step1_epoch{args.epochs_step1}_step2_epoch{epoch}.pt'):
            loralib.load_lora(model, save_dir + f'step1_epoch{args.epochs_step1}_step2_epoch{epoch}.pt')
            optimizer.load_state_dict(torch.load(save_dir + f'step1_epoch{args.epochs_step1}_step2_epoch{epoch}_op.pt'))
            iteration = epoch * len(train_loader)
            continue
        
        for data in tqdm(train_loader, f'Step2 Epoch: {epoch:03d}'):
            images, attrs, _ = data
            
            outputs = model(images.to("cuda"))
            cls_loss = cls_loss_fn(outputs, attrs[:,0].to("cuda"))
            
            all_keys = [k for k in all_features.keys() if "lora0" in k]
            all_features1, all_features2 = [], []
            for k in all_keys:
                all_features1.append(all_features[k].detach())
                all_features2.append(all_features[k.replace("lora0", "lora1")])
            ortho_loss = ortho_loss_fn(all_features1, all_features2, args.l1)
            
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
                    "step2/iter": iteration,
                    "step2/loss": train_loss,
                    "step2/loss_cls": train_cls_loss,
                    "step2/loss_ortho": train_ortho_loss,
                    "step2/train_worst_acc": train_worst_acc,
                    "step2/train_avg_acc": train_avg_acc,
                })

                print(f'\nIteration: {iteration:06d}, LR: {lr:.06f}, L: {train_loss:.03f}, L_cls: {train_cls_loss:.03f}, L_ortho: {train_ortho_loss:.03f}')

                train_losses, train_cls_losses, train_ortho_losses = [], [], []
                train_preds, train_labels, train_spurious = [], [], []

        loralib.save_lora(model, save_dir + f'step1_epoch{args.epochs_step1}_step2_epoch{epoch}.pt', idxs=[1])
        torch.save(optimizer.state_dict(), save_dir + f'step1_epoch{args.epochs_step1}_step2_epoch{epoch}_op.pt')
    
    wandb.finish()