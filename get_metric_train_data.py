
import json
from metric.perspective_api import PerspectiveApiScorer
from metric.detoxify import DetoxifyScorer
import argparse
from tqdm import tqdm
import pdb
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

api_key = open(".perspective_api_key").read().strip()
perspective_scorer = PerspectiveApiScorer(api_key)
detoxify_scorer = DetoxifyScorer()

def map_values_to_binary(df, threshold=0.5, use_binary=True):
    # map to discrete values
    def map_value(x):
        return 1 if x > threshold else 0
    if use_binary:
        df = df.applymap(map_value)

    # Compute the 'Any' Value
    def check_any_ones(row):
        return 1 if (row > 0).any() else 0
    any_col = df.apply(check_any_ones, axis=1)

    # Compute the 'All' Value
    def check_all(row):
        return row.sum()
    all_col = df.apply(check_all, axis=1)
    
    df['any'] = any_col
    df['all'] = all_col
    return df

def float_it(score):
    for key in score.keys():
        score[key] = float(score[key])
    return score


file = open("harmful_corpus/derogatory_corpus.csv", "r")
# file = open("harmful_corpus/advbench_harmful_strings.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
# num=2 # debug
for i in range(num):
    targets.append(data[i][0])

# Detoxify Results
detoxify_res = detoxify_scorer.get_scores(targets)
detoxify_res = pd.DataFrame(detoxify_res)
detoxify_res_origin = detoxify_res.copy()
# Draw Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(detoxify_res, annot=False, cmap='coolwarm', fmt='g', yticklabels=False)
plt.title("Detoxify Heatmap for Training Corpus")
plt.tight_layout()
plt.savefig('harmful_corpus/detoxify_heatmap.png')
# Draw Barplot
detoxify_res = map_values_to_binary(detoxify_res)
plt.figure(figsize=(5, 4))
detoxify_res.sum().plot(kind='bar')
plt.title("Detoxify Scores for Training Corpus")
plt.tight_layout()
plt.savefig('harmful_corpus/detoxify_scores.png')
detoxify_res.to_csv("harmful_corpus/detoxify_res.csv", index=False)
# Rank Discreate
print('Detoxify Rank Discreate:', detoxify_res.sort_values(by='all',ascending=False).index.tolist())
# Rank Continuous
detoxify_res = map_values_to_binary(detoxify_res_origin, threshold=0.5, use_binary=False)
print('Detoxify Rank Continuous:', detoxify_res.sort_values(by='all',ascending=False).index.tolist())


# Perspective API Results
perspective_res = []
for target in targets:
    perspective_res.append(float_it(perspective_scorer.get_scores(target)))
perspective_res = pd.DataFrame(perspective_res)
perspective_res_origin = perspective_res.copy()
# Draw Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(perspective_res, annot=False, cmap='coolwarm', fmt='g', yticklabels=False)
plt.title("Perspective Heatmap for Training Corpus")
plt.tight_layout()
plt.savefig('harmful_corpus/perspective_heatmap.png')

# Draw Barplot
perspective_res = map_values_to_binary(perspective_res)
plt.figure(figsize=(5, 4))
perspective_res.sum().plot(kind='bar')
plt.title("Perspective Scores for Training Corpus")
plt.tight_layout()
plt.savefig('harmful_corpus/perspective_scores.png')
perspective_res.to_csv("harmful_corpus/perspective_res.csv", index=False)

# Rank Discreate
print('perspective Rank Discreate:', perspective_res.sort_values(by='all',ascending=False).index.tolist())
# Rank Continuous
perspective_res = map_values_to_binary(perspective_res_origin, threshold=0.5, use_binary=False)
print('perspective Rank Continuous:', perspective_res.sort_values(by='all',ascending=False).index.tolist())

# detoxify_res.to_csv("harmful_corpus/detoxify_detoxify_res.csv", index=False)
