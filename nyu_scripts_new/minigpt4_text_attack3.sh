#!/bin/bash
export HUGGINGFACE_HUB_CACHE=/scratch/zx1673/codes/jailbreak_proj/model 
export TORCH_HOME=/scratch/zx1673/codes/jailbreak_proj/model 


cd /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models


python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 10 \
                        --select_method Detoxify_Rank_Discreate \

python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 20 \
                        --select_method Detoxify_Rank_Discreate \

python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 30 \
                        --select_method Detoxify_Rank_Discreate \

python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 40 \
                        --select_method Detoxify_Rank_Discreate \

python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 50 \
                        --select_method Detoxify_Rank_Discreate \



# Rever
python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 10 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select

python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 20 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select

python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 30 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select

python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 40 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select

python minigpt_textual_attack.py --cfg_path eval_configs/minigpt4_eval.yaml  --gpu_id 0 \
                        --n_iters 5000 --batch_size 8 --n_candidates 50 \
                        --save_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results \
                        --llm_model vicuna_7b \
                        --corpus_selection_num 50 \
                        --select_method Detoxify_Rank_Discreate \
                        --rever_select