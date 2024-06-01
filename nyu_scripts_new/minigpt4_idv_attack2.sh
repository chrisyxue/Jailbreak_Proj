#!/bin/bash
export HUGGINGFACE_HUB_CACHE=/scratch/zx1673/codes/jailbreak_proj/model 
export TORCH_HOME=/scratch/zx1673/codes/jailbreak_proj/model 


cd /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models

python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 1000 --constrained --eps 32 --alpha 1 \
                        --batch_size 8 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --attack_individual_corpus