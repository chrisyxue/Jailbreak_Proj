#!/bin/bash

# conda activate minigpt4

# PerspectiveAPIkey: AIzaSyBlq-DJchJ7Af5bbS8OIKtDQBu4tKxaDxc
export HUGGINGFACE_HUB_CACHE=/scratch/zx1673/codes/jailbreak_proj/model 
export TORCH_HOME=/scratch/zx1673/codes/jailbreak_proj/model 

# Generate visual adversarial examples on MiniGPT-4 (Vicuna-13B)
python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 500 --constrained --eps 16 --alpha 1 \
                        --batch_size 8 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b

# python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
#                         --n_iters 5000 --constrained --eps 16 --alpha 1 \
#                         --batch_size 8 \
#                         --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
# # python get_metric_train_data.py