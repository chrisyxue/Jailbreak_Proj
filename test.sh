#!/bin/bash

# conda activate minigpt4

# PerspectiveAPIkey: AIzaSyBlq-DJchJ7Af5bbS8OIKtDQBu4tKxaDxc
export HUGGINGFACE_HUB_CACHE=/scratch/zx1673/codes/jailbreak_proj/model 
export TORCH_HOME=/scratch/zx1673/codes/jailbreak_proj/model 

# Generate visual adversarial examples on MiniGPT-4 (Vicuna-13B)
# python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
#                         --n_iters 50 --constrained --eps 16 --alpha 0.5 \
#                         --batch_size 8 \
#                         --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
#                         --subset_corpus_num 10

# python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
#                         --n_iters 50 --constrained --eps 16 --alpha 0.5 \
#                         --batch_size 8 \
#                         --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
#                         --subset_corpus_num 20

# python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
#                         --n_iters 50 --constrained --eps 16 --alpha 0.5 \
#                         --batch_size 8 \
#                         --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
#                         --subset_corpus_num 30

# python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
#                         --n_iters 50 --constrained --eps 16 --alpha 0.5 \
#                         --batch_size 8 \
#                         --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
#                         --subset_corpus_num 40

# python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
#                         --n_iters 50 --constrained --eps 16 --alpha 0.5 \
#                         --batch_size 8 \
#                         --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
#                         --subset_corpus_num 50


## Data Selection

python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 1 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b