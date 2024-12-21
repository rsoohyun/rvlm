import os
import glob
import argparse
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from data import load_dataset
from core import clip, loralib, utils
from utils import *



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
    parser.add_argument("--last_only", action="store_true")
    
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@SepLoRA@r4")
    parser.add_argument("--sim_dir", type=str, default="./experiments/desc_sim")
    
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--org", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    
    
    if args.infer:
        device = "cuda" if torch.cuda.is_available() else "cpu"    
        utils.set_seed(args.seed)
        
        lora_idxs = list(range(args.num_lora))
        lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]
        
        lm, r = args.save_dir.replace('@wp', '').split('@')[-2:]
        if args.r != int(r[1:]):
            raise NotImplementedError('Please match configuration "r".')
        if set(list(lm.split('_'))) != set(lora_modules):
            raise NotImplementedError('Please match configuration "lora_modules".')
        
        ## Load data and model
        if args.arch == "CLIP":
            model = clip.CLIP_FT("ViT-L/14", device, n_cls=args.n_cls)
            model.eval()
        else:
            raise NotImplementedError(f'{args.arch} is not implemented yet.')
        model.eval()
        
        ## Load data and model
        test_dataset = load_dataset(args.data_dir, args.dataset, "test", model.preprocess, args.prompt_id)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
        
        desc_feats = model.model.encode_text(clip.tokenize(test_dataset.all_descs).to(device))
        desc_feats = desc_feats / desc_feats.norm(dim=1, keepdim=True)
        
        def extract_desc(model, sim_dir):
            os.makedirs(sim_dir, exist_ok=True)
            
            all_features = {}
            def get_output(name):
                def hook(model, input, output):
                    all_features[name] = output
                return hook
            
            for name, submodule in model.model.visual.transformer.resblocks.named_modules():
                idx = name.split('.')[0]
                if isinstance(submodule, clip.model.ResidualAttentionBlock):
                    eval(f"model.model.visual.transformer.resblocks[{idx}]").register_forward_hook(get_output(name))
            
            for idx, data in enumerate(tqdm(test_loader, desc="extracting desc sim")):
                images, attrs, _ = data
                labels = attrs[:,0]
                spurious = attrs[:,1]
                
                outputs = model(images.to(device)).detach().cpu()
                _, preds = torch.max(outputs, 1)
                
                tmp_features = defaultdict(list)
                if args.last_only: all_keys = ["23"]
                else: all_keys = list(all_features.keys())
                
                for i in lora_idxs:
                    loralib.set_used_lora(model, [i])
                    model(images.to(device))
                    img_feats = [all_features[k] for k in all_keys]
                    img_feats = [model.model.visual.early_exit_proj(feat) for feat in img_feats]
                    img_feats = [feat / feat.norm(dim=-1, keepdim=True) for feat in img_feats]
                    tmp_features[i] = [(feat @ desc_feats.t()).detach().cpu() for feat in img_feats]
                loralib.set_used_lora(model, lora_idxs)
                
                result = {
                    "features": tmp_features,
                    "preds": preds,
                    "labels": labels,
                    "spurious": spurious,
                }
                torch.save(result, f"{sim_dir}/{idx}.pt")
        
        if args.org:
            extract_desc(model, f"{args.sim_dir}/CLIP")
            
        loralib.apply_lora(model, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules=lora_modules)
        loralib.load_lora(model, args.save_dir + f'/epoch{args.epochs}.pt', device=torch.device(device))
        loralib.set_used_lora(model, lora_idxs)
        model.eval()
        extract_desc(model, f"{args.sim_dir}/{args.save_dir.split('/')[-1]}")


    if args.vis:
        sim_dir = f"{args.sim_dir}/{args.save_dir.split('/')[-1]}"
        save_dir = sim_dir.replace('desc_sim/', 'desc_sim_vis/')
        os.makedirs(save_dir, exist_ok=True)
        
        prompt_id = int(sim_dir.split('_p')[-1][0])
        classes = ['landbird', 'waterbird']
        class_descs = read_json(f"{args.data_dir}/{args.dataset}/descs{prompt_id}.json")
        all_descs = list(set(itertools.chain(*list(class_descs.values()))))
        
        def find_set(desc):
            cls0 = desc in class_descs['landbird']
            cls1 = desc in class_descs['waterbird']
            if cls0 and cls1: return 2
            elif cls0: return 0
            elif cls1: return 1
        
        set_list = [find_set(desc) for desc in all_descs]
        idxs = np.argsort(set_list)
        color_dict = {0:'r', 1:'b', 2:'g'}
        color_list = [color_dict[set_list[i]] for i in idxs]
        all_descs = [all_descs[i] for i in idxs]
        
        result_paths = glob.glob(f"{sim_dir}/*.pt")
        for path in tqdm(result_paths):
            idx = path.split('/')[-1][:-3]
            data = torch.load(path)
            features = data["features"]
            preds = data["preds"]
            labels = data["labels"]
            spurious = data["spurious"]
            
            correct = preds == labels
            sims = [features[i][-1][:, idxs] for i in range(args.num_lora)]  # [bsz, num_desc]
            
            for i in range(sims[0].shape[0]):
                tmp_sims = [sim[i] for sim in sims]
                y_min = min([sim.min().item() for sim in sims])
                y_max = max([sim.max().item() for sim in sims])
                
                fig, axs = plt.subplots(2, 2)
                for j, sim in enumerate(tmp_sims):
                    data = pd.Series(
                        sim.tolist(),
                        index=all_descs
                    )
                    data.plot( 
                        kind='bar', 
                        ax = axs[j//2, j%2],
                        color=color_list,
                    )
                    axs[j//2, j%2].set_title(f"lora {j}")
                    axs[j//2, j%2].set_ylim(y_min, y_max)
                    axs[j//2, j%2].tick_params(axis='x', labelsize=2.5)
                
                for ax in fig.get_axes():
                    ax.label_outer()
                
                title = f"{'correct' if correct[i] else 'wrong'} | {'landbird' if labels[i]==0 else 'waterbird'} | {'land' if spurious[i]==0 else 'water'}"
                fig.suptitle(title)
                plt.subplots(constrained_layout=True)
                plt.tight_layout()
                fig.savefig(f"{save_dir}/{title.replace(' | ', '_')}_{idx}_{i}.png", dpi=250)
                plt.close()