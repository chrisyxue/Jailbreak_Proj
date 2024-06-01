#!/bin/bash

# conda activate minigpt4
export HUGGINGFACE_HUB_CACHE=/scratch/zx1673/codes/jailbreak_proj/model 
export TORCH_HOME=/scratch/zx1673/codes/jailbreak_proj/model 

# Eval Given Image on RealToxicityPrompts
python get_metric.py \
        --input /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_10_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/rtp_results.jsonl \
        --output /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_10_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/result_eval.jsonl \
        --output_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_10_Detoxify_Rank_Continuous/attack_5000_16_1.0_8

python get_metric.py \
        --input /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_20_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/rtp_results.jsonl \
        --output /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_20_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/result_eval.jsonl \
        --output_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_20_Detoxify_Rank_Continuous/attack_5000_16_1.0_8

python get_metric.py \
        --input /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_30_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/rtp_results.jsonl \
        --output /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_30_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/result_eval.jsonl \
        --output_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_30_Detoxify_Rank_Continuous/attack_5000_16_1.0_8

python get_metric.py \
        --input /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_40_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/rtp_results.jsonl \
        --output /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_40_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/result_eval.jsonl \
        --output_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_40_Detoxify_Rank_Continuous/attack_5000_16_1.0_8

python get_metric.py \
        --input /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_50_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/rtp_results.jsonl \
        --output /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_50_Detoxify_Rank_Continuous/attack_5000_16_1.0_8/result_eval.jsonl \
        --output_dir /scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_selection_50_Detoxify_Rank_Continuous/attack_5000_16_1.0_8