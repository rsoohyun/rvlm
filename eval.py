import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_dataset
from core import clip, loralib, utils
import re

# https://github.com/chingyaoc/debias_vl.git
def infer(model, data_loader, desc=''):
    all_preds = []
    all_labels = []
    all_spurious = []
    for data in tqdm(data_loader, desc=desc):
        images, attrs, _ = data
        images = images.to(model.device)
        labels = attrs[:,0]
        spurious = attrs[:,1]
        
        outputs = model(images).detach().cpu()
        _, preds = torch.max(outputs, 1)
        
        all_preds.append(preds)
        all_labels.append(labels)
        all_spurious.append(spurious)
        
    all_preds, all_labels, all_spurious = torch.concat(all_preds, dim=0).numpy(), torch.concat(all_labels, dim=0).numpy(), torch.concat(all_spurious, dim=0).numpy()
    return all_preds, all_labels, all_spurious

def evaluate(all_preds, all_labels, all_spurious):
    correct_by_group = [[0, 0], [0, 0]]
    total_by_group   = [[0, 0], [0, 0]]
    accs_by_group    = [[0, 0], [0, 0]]
    correct = all_preds == all_labels
    
    for t in [0, 1]:
        for s in [0 ,1]:
            ix = np.where(np.logical_and(all_labels == t, all_spurious == s))[0]
            correct_by_group[t][s] += np.sum(correct[ix])
            total_by_group[t][s] += len(ix)
            accs_by_group[t][s] = np.sum(correct[ix]) / len(ix)
        
    # Average accuracy
    avg_acc = (
        correct_by_group[0][0] +
        correct_by_group[0][1] +
        correct_by_group[1][0] +
        correct_by_group[1][1]
    )
    avg_acc = avg_acc * 100 / np.sum(np.array(total_by_group))
    
    accs_by_group = np.array(accs_by_group).flatten() * 100
    worst_acc = np.min(accs_by_group)
    
    del all_preds
    
    return worst_acc, avg_acc, accs_by_group.tolist()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP")
    parser.add_argument("--dataset", type=str, default="waterbird")
    
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--lora_mlp", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epochs_step1", type=int, default=4)
    parser.add_argument("--epochs_step2", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@SepLoRA@r4")
    
    parser.add_argument("--eval_org", action="store_true")
    args = parser.parse_args()
    
    utils.set_seed(args.seed)
    args.r = int(args.save_dir[:-1].split('@r')[-1])
    args.lora_mlp = "@mlp" in args.save_dir
    
    ## Load data and model
    test_dataset = load_dataset(args.data_dir, args.dataset, "test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    
    if args.arch == "CLIP":
        model = clip.CLIP_FT("ViT-L/14", "cuda", n_cls=test_dataset.n_classes)
    else:
        raise NotImplementedError(f'{args.arch} is not implemented yet.')
    
    print(f"Evaluation on \"{args.save_dir}\"")
    
    ## Evaluation
    if args.eval_org:
        model.eval()
        worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, desc="Eval CLIP"))
        accs_by_group_str = f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]"
        print("== CLIP ==")
        print(f"Average accuracy: {avg_acc:.2f}")
        print(f"Worst Group accuracy: {worst_acc:.2f}")
        print(f"Acc by group: {accs_by_group_str}")
    
    if '@SepLoRA' in args.save_dir:
        loralib.apply_lora(model, 2, args.r, args.lora_alpha, args.lora_dropout, mlp=args.lora_mlp)
        
        # lora1 eval
        loralib.load_lora(model, args.save_dir + f'/step1_epoch{args.epochs_step1}.pt')
        loralib.set_used_lora(model, [0])
        model.eval()
        
        worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, desc="Eval CLIP+LoRA1"))
        accs_by_group_str = f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]"
        print("== Step1) CLIP+LoRA1 ==")
        print(f"Average accuracy: {avg_acc:.2f}")
        print(f"Worst Group accuracy: {worst_acc:.2f}")
        print(f"Acc by group: {accs_by_group_str}")
        
        # lora2 eval
        loralib.load_lora(model, args.save_dir + f'/step1_epoch{args.epochs_step1}_step2_epoch{args.epochs_step2}.pt')
        loralib.set_used_lora(model, [1])
        model.eval()

        worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, desc="Eval CLIP+LoRA2"))
        accs_by_group_str = f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]"
        print("== Step2) CLIP+LoRA2 ==")
        print(f"Average accuracy: {avg_acc:.2f}")
        print(f"Worst Group accuracy: {worst_acc:.2f}")
        print(f"Acc by group: {accs_by_group_str}")

    # MultiLoRA
    elif "@MultiLoRA" in args.save_dir:
        print("Evaluating MultiLoRA...")

        num_lora_match = re.search(r"@num_lora(\d+)", args.save_dir)
        if num_lora_match:
            num_lora = int(num_lora_match.group(1))
        else:
            raise ValueError("cannot find num_lora in save_dir")

        loralib.apply_lora(model, num_lora, args.r, args.lora_alpha, args.lora_dropout, mlp=False)
        loralib.load_lora(model, args.save_dir + f"/epoch{args.num_lora}.pt")  #

        if "gating" in args.save_dir:  # save_dir에 'gating' 포함 여부 확인
            print("Using gating mechanism for evaluation.")
            loralib.set_used_lora(model, list(range(args.num_lora)))  
        else:
            print("Using cumulative sum of all LoRA components.")
            loralib.set_used_lora(model, list(range(args.num_lora))) 

        model.eval()
        worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, desc="Eval CLIP+MultiLoRA"))
        accs_by_group_str = f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]"
        print("== CLIP+MultiLoRA ==")
        print(f"Average accuracy: {avg_acc:.2f}")
        print(f"Worst Group accuracy: {worst_acc:.2f}")
        print(f"Acc by group: {accs_by_group_str}")