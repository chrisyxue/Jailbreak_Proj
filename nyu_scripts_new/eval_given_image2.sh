#!/bin/bash

# conda activate minigpt4

export HUGGINGFACE_HUB_CACHE=/scratch/zx1673/codes/jailbreak_proj/model 
export TORCH_HOME=/scratch/zx1673/codes/jailbreak_proj/model
cd /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models


# eps 16


python minigpt_inference.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 \
        --image_file adversarial_images/prompt_constrained_16.bmp \
        --n_iters 5000 --constrained --eps 16 --alpha 1 \
        --batch_size 8 \
        --data_path /scratch/zx1673/codes/jailbreak_proj/dataset/realtoxicityprompts-data/prompts.jsonl \
        --llm_model vicuna_7b \
        --corpus_selection_num 40 \
        --select_method Detoxify_Rank_Discreate

python minigpt_inference.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 \
        --image_file adversarial_images/prompt_constrained_16.bmp \
        --n_iters 5000 --constrained --eps 16 --alpha 1 \
        --batch_size 8 \
        --data_path /scratch/zx1673/codes/jailbreak_proj/dataset/realtoxicityprompts-data/prompts.jsonl \
        --llm_model vicuna_7b \
        --corpus_selection_num 50 \
        --select_method Detoxify_Rank_Discreate


python minigpt_inference.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 \
        --image_file adversarial_images/prompt_constrained_16.bmp \
        --n_iters 5000 --constrained --eps 32 --alpha 1 \
        --batch_size 8 \
        --data_path /scratch/zx1673/codes/jailbreak_proj/dataset/realtoxicityprompts-data/prompts.jsonl \
        --llm_model vicuna_7b \
        --corpus_selection_num 40 \
        --select_method Detoxify_Rank_Discreate

python minigpt_inference.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 \
        --image_file adversarial_images/prompt_constrained_16.bmp \
        --n_iters 5000 --constrained --eps 32 --alpha 1 \
        --batch_size 8 \
        --data_path /scratch/zx1673/codes/jailbreak_proj/dataset/realtoxicityprompts-data/prompts.jsonl \
        --llm_model vicuna_7b \
        --corpus_selection_num 50 \
        --select_method Detoxify_Rank_Discreate