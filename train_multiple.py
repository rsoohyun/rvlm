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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP")
    parser.add_argument("--dataset", type=str, default="waterbird")
    parser.add_argument("--n_cls", type=int, default=2)

    parser.add_argument("--num_lora", type=int, default=4)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_feature_ortho", type=float, default=0.0)
    parser.add_argument("--lambda_param_ortho", type=float, default=0.0)
    parser.add_argument("--l1", action="store_true")

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@MultiLoRA")
    parser.add_argument("--resume_id", type=str, default="")

    parser.add_argument("--use_gating", action="store_true")

    args = parser.parse_args()

    # Set ENV
    utils.set_seed(args.seed)
    save_dir = args.save_dir
    save_dir += f"@numlora_{args.num_lora}" 
    save_dir += f"@feature_{args.lambda_feature_ortho}"
    save_dir += f"@param_{args.lambda_param_ortho}"
    if args.use_gating:
        save_dir += "@gating"

    save_dir += f"@r{args.r}/"
    os.makedirs(save_dir, exist_ok=True)

    if args.resume_id:
        wandb.init(project="rvlm", id=args.resume_id, resume=True)
    else:
        wandb.init(project="rvlm")
    wandb.config.update(args)
    wandb.run.name = save_dir.split('/')[-2]

    # Load data and model
    if args.arch == "CLIP":
        model = clip.CLIP_FT("ViT-L/14", "cuda", n_cls=args.n_cls)
    else:
        raise NotImplementedError(f"{args.arch} is not implemented yet.")

    print(f"{args.arch} w/o LoRA: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    loralib.apply_lora(model, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, mlp=True, use_gating=args.use_gating)
    print(f"{args.arch} w/ LoRA: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


    
    train_dataset = load_dataset(args.data_dir, args.dataset, "train", model.preprocess)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    # Loss functions
    cls_loss_fn = nn.CrossEntropyLoss()
    feature_ortho_loss_fn = losses.OrthogonalLoss()
    param_ortho_loss_fn = losses.ParameterOrthogonalityLoss()

    # Hook for collecting features
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

    # Optimizer
    _, trainable_params = loralib.get_lora_params(model, fc=True, idxs=list(range(args.num_lora)))
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    # Training loop
    iteration = 0
    for epoch in range(1, args.epochs + 1):
        train_losses, train_cls_losses, train_feature_ortho_losses, train_param_ortho_losses = [], [], [], []
        train_preds, train_labels, train_spurious = [], [], []

        for data in tqdm(train_loader, desc=f"Epoch {epoch:03d}"):
            images, attrs, _ = data

            # Forward pass
            outputs = model(images.to("cuda"))
            cls_loss = cls_loss_fn(outputs, attrs[:, 0].to("cuda"))


            # Feature orthogonality loss
            feature_ortho_loss = 0.0
            if args.lambda_feature_ortho > 0.0:
                #feature_list = [all_features[k] for k in sorted(all_features.keys())]
                feature_ortho_loss = feature_ortho_loss_fn(all_features, args.num_lora, args.l1)

            # Parameter orthogonality loss
            param_ortho_loss = 0.0
            if args.lambda_param_ortho > 0.0:
                param_ortho_loss = param_ortho_loss_fn(model)

            # Combined loss
            loss = (
                args.lambda_cls * cls_loss
                + args.lambda_feature_ortho * feature_ortho_loss
                + args.lambda_param_ortho * param_ortho_loss
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log losses
            train_losses.append(loss.item())
            train_cls_losses.append(cls_loss.item())
            train_feature_ortho_losses.append(feature_ortho_loss if isinstance(feature_ortho_loss, float) else feature_ortho_loss.item())
            train_param_ortho_losses.append(param_ortho_loss if isinstance(param_ortho_loss, float) else param_ortho_loss.item())
            _, preds = torch.max(outputs, 1)
            train_preds.append(preds)
            train_labels.append(attrs[:, 0])
            train_spurious.append(attrs[:, 1])

            iteration += 1

            if iteration % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                train_loss = np.mean(train_losses)
                train_cls_loss = np.mean(train_cls_losses)
                train_feature_ortho_loss = np.mean(train_feature_ortho_losses)
                train_param_ortho_loss = np.mean(train_param_ortho_losses)

                train_preds = torch.cat(train_preds).cpu().numpy()
                train_labels = torch.cat(train_labels).cpu().numpy()
                train_spurious = torch.cat(train_spurious).cpu().numpy()
                train_worst_acc, train_avg_acc, _ = evaluate(train_preds, train_labels, train_spurious)

                wandb.log(
                    {
                        "iter": iteration,
                        "loss": train_loss,
                        "loss_cls": train_cls_loss,
                        "loss_feature_ortho": train_feature_ortho_loss,
                        "loss_param_ortho": train_param_ortho_loss,
                        "train_worst_acc": train_worst_acc,
                        "train_avg_acc": train_avg_acc,
                    }
                )

                print(
                    f"Iter {iteration:06d}, LR: {lr:.6f}, Loss: {train_loss:.3f}, Loss_cls: {train_cls_loss:.3f}, Loss_feature_ortho: {train_feature_ortho_loss:.3f}, Loss_param_ortho: {train_param_ortho_loss:.3f}"
                )

                train_losses, train_cls_losses, train_feature_ortho_losses, train_param_ortho_losses = [], [], [], []
                train_preds, train_labels, train_spurious = [], [], []

        # Save checkpoint
        loralib.save_lora(model, os.path.join(save_dir, f"epoch{epoch}.pt"), idxs=list(range(args.num_lora)))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, f"epoch{epoch}_opt.pt"))

    wandb.finish()
