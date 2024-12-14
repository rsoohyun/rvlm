import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_dataset
from core import clip, loralib, utils


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
    parser.add_argument("--n_cls", type=int, default=2)
    parser.add_argument("--prompt_id", type=int, default=0)
    
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--num_lora", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=4.)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--lora_modules", type=str, default="q,v")
    
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epochs_per_step", type=str, default="4,4")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@SepLoRA@r4")
    
    parser.add_argument("--eval_org", action="store_true")
    args = parser.parse_args()
    
    f = open('./eval_log.txt', 'a')
    
    utils.set_seed(args.seed)
    
    lora_idxs = list(range(args.num_lora))
    lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]
    
    lm, r = args.save_dir.replace('@wp', '').split('@r')[-2:]
    if args.r != int(r):
        raise NotImplementedError('Please match configuration "r".')
    if set(list(lm.split('_'))) != set(lora_modules):
        raise NotImplementedError('Please match configuration "lora_modules".')
    
    ## Load data and model
    if args.arch == "CLIP":
        model = clip.CLIP_FT("ViT-L/14", "cuda", n_cls=args.n_cls)
        model.eval()
    else:
        raise NotImplementedError(f'{args.arch} is not implemented yet.')
    
    test_dataset = load_dataset(args.data_dir, args.dataset, "test", model.preprocess, args.prompt_id)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    
    f.write(f"Evaluation on \"{args.save_dir}\"\n")
    
    ## Evaluation
    if args.eval_org:
        worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, desc="Eval CLIP"))
        accs_by_group_str = f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]"
        f.write("== CLIP ==\n")
        f.write(f"Average accuracy: {avg_acc:.2f}\n")
        f.write(f"Worst Group accuracy: {worst_acc:.2f}\n")
        f.write(f"Acc by group: {accs_by_group_str}\n")
    
    if '@LoRA' in args.save_dir:
        loralib.apply_lora(model, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules=lora_modules)
        loralib.load_lora(model, args.save_dir + f'/epoch{args.epochs}.pt')
        loralib.set_used_lora(model, lora_idxs)
        
        worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, desc="Eval CLIP+LoRA"))
        accs_by_group_str = f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]"
        f.write("== CLIP+LoRA ==\n")
        f.write(f"Average accuracy: {avg_acc:.2f}\n")
        f.write(f"Worst Group accuracy: {worst_acc:.2f}\n")
        f.write(f"Acc by group: {accs_by_group_str}\n")
    
    elif '@SepLoRA' in args.save_dir:
        loralib.apply_lora(model, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules=lora_modules)
        
        train_epochs = [int(m) for m in args.epochs_per_step.split(',')]
        if len(train_epochs) != args.num_lora:
            raise NotImplementedError('Wrong number of training steps.')
        
        for i in range(args.num_lora):
            loralib.load_lora(model, args.save_dir + f'/step{i+1}_epoch{train_epochs[i]}.pt')
            loralib.set_used_lora(model, [i])
            
            worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, desc=f"Eval CLIP+LoRA{i+1}"))
            accs_by_group_str = f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]"
            f.write(f"== Step{i+1}) CLIP+LoRA{i+1} ==\n")
            f.write(f"Average accuracy: {avg_acc:.2f}\n")
            f.write(f"Worst Group accuracy: {worst_acc:.2f}\n")
            f.write(f"Acc by group: {accs_by_group_str}\n")
        
    f.write('\n')
    f.close()