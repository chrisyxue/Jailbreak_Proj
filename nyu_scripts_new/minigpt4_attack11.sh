#!/bin/bash
export HUGGINGFACE_HUB_CACHE=/scratch/zx1673/codes/jailbreak_proj/model 
export TORCH_HOME=/scratch/zx1673/codes/jailbreak_proj/model 

cd /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models

python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --constrained --eps 16 --alpha 1 \
                        --batch_size 8 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --corpus_selection_num 10 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select

python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --constrained --eps 16 --alpha 1 \
                        --batch_size 8 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --corpus_selection_num 20 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select

python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --constrained --eps 16 --alpha 1 \
                        --batch_size 8 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --corpus_selection_num 30 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select

python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --constrained --eps 16 --alpha 1 \
                        --batch_size 8 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --corpus_selection_num 40 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select

python minigpt_visual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --constrained --eps 16 --alpha 1 \
                        --batch_size 8 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --corpus_selection_num 50 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select