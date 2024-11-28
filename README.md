# Robust VLM

### Train
- Train MHA only
    ```bash
    python train.py --r 4 --lora_alpha 1 --lora_dropout 0. --epochs_step1 4 --epochs_step2 4 --batch_size 32 --lr 1e-4 --wd 5e-5 --save_dir "./experiments/models/CLIP@SepLoRA"
    ```
- Train MHA + MLP (add `--lora_mlp`)
    ```bash
    python train.py --lora_mlp --r 4 --lora_alpha 1 --lora_dropout 0. --epochs_step1 4 --epochs_step2 4 --batch_size 32 --lr 1e-4 --wd 5e-5 --save_dir "./experiments/models/CLIP@SepLoRA"
    ```

### Evaluate
```bash
python eval.py --save_dir "./experiments/models/CLIP@SepLoRA@4"
```