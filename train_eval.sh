#!/bin/bash

# Parse arguments for GPU ID, num_lora, and r
GPU=$1
NUM_LORA=$2
R=$3

# Common variables
BATCH_SIZE=32
SAVE_BASE_DIR="./experiments/models"
SAVE_DIR="${SAVE_BASE_DIR}/CLIP@MultiLoRA"

# Lambda values
LAMBDA_CLS=1.0
LAMBDA_FEATURE_VALUES=(0.0 1.0)
LAMBDA_PARAM_VALUES=(0.0 1.0)
USE_GATING_VALUES=(false true)

# Function: Run training and evaluation
run_experiment() {
    LAMBDA_FEATURE=$1
    LAMBDA_PARAM=$2
    USE_GATING=$3

    # Create save directory
    SAVE_DIR="${SAVE_BASE_DIR}/CLIP@MultiLoRA"
    SAVE_DIR+="@feature_${LAMBDA_FEATURE}"
    SAVE_DIR+="@param_${LAMBDA_PARAM}"
    if [ "$USE_GATING" = true ]; then
        SAVE_DIR+="@gating"
    fi
    SAVE_DIR+="@r${R}/"

    # Train
    CUDA_VISIBLE_DEVICES=$GPU python train_multiple.py \
        --num_lora $NUM_LORA \
        --r $R \
        --batch_size $BATCH_SIZE \
        --lambda_feature_ortho $LAMBDA_FEATURE \
        --lambda_param_ortho $LAMBDA_PARAM \
        $(if [ "$USE_GATING" = true ]; then echo "--use_gating"; fi)

    # Evaluate
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --save_dir $SAVE_DIR
}

# Main loop: Run all combinations
for LAMBDA_FEATURE in "${LAMBDA_FEATURE_VALUES[@]}"; do
    for LAMBDA_PARAM in "${LAMBDA_PARAM_VALUES[@]}"; do
        for USE_GATING in "${USE_GATING_VALUES[@]}"; do
            run_experiment $LAMBDA_FEATURE $LAMBDA_PARAM $USE_GATING
        done
    done
done

echo "All experiments completed."
