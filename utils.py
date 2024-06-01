import os
import glob
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import pdb

def render_save_dir(args,text_attack=False):
    save_dir = args.save_dir
    
    if text_attack:
        save_dir = save_dir + '_text_attack'
    if args.llm_model == 'vicuna_7b':
        save_dir = save_dir + '_vicuna_7b'

    if text_attack==False and args.constrained:
        save_dir = os.path.join(save_dir, f'constrained')
    if args.attack_individual_corpus:
        save_dir = os.path.join(save_dir, f'attack_individual_corpus')
    if args.subset_corpus_num > 0:
        save_dir = os.path.join(save_dir, f'attack_subset_{args.subset_corpus_num}')
    if args.corpus_selection_num > 0:
        save_dir = os.path.join(save_dir, f'attack_selection_{args.corpus_selection_num}_{args.select_method}')
        if args.rever_select:
            save_dir = os.path.join(save_dir, f'rever_select')
    if text_attack == True:
        save_dir = os.path.join(save_dir, f'attack_{args.n_iters}_{args.n_candidates}_{args.batch_size}')
    else:
        save_dir = os.path.join(save_dir, f'attack_{args.n_iters}_{args.eps}_{args.alpha}_{args.batch_size}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Save Dir: ' + save_dir)
    return save_dir

def loss_std(loss, window=20):
    # Take the last 'window' number of loss
    recent_loss = loss[-window:]
    stability = np.std(recent_loss)
    return stability

def loss_mean(loss, window=20):
    # Take the last 'window' number of loss
    recent_loss = loss[-window:]
    avg = np.mean(recent_loss)
    return avg


#  torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))

def read_idv_attack_loss_file(base_dir,window=20):
    # read copurs dir
    corpus_dirs = [i for i in os.listdir(base_dir) if 'corpus' in i]
    corpus_dirs_id = [int(i.split('_')[-1]) for i in corpus_dirs]
    sorted_indices = [index for index, _ in sorted(enumerate(corpus_dirs_id), key=lambda pair: pair[1])]
    corpus_dirs = [corpus_dirs[i] for i in sorted_indices]
    loss_lst = []
    for dir in corpus_dirs:
        dir = os.path.join(base_dir, dir)
        loss = torch.load(os.path.join(dir, 'loss'))
        loss_lst.append(loss)
    
    # Draw the figure for loss trajectory
    grid_size = math.ceil(math.sqrt(len(loss_lst)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axs = axs.flatten()  # flatten the axes array
    for i, loss in enumerate(loss_lst):
        sns.lineplot(data=loss, ax=axs[i])
        axs[i].set_title(f'Loss for {corpus_dirs[i]}')
        axs[i].grid(True)
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.savefig('test.png')
    plt.close(fig)


    # Draw the figure for loss mean and std in a window size.
    loss_mean_value_lst, loss_std_value_lst = [], []
    for i, loss in enumerate(loss_lst):
        loss_mean_value = loss_mean(loss, window=window)
        loss_std_value = loss_std(loss, window=window)
        loss_mean_value_lst.append(loss_mean_value)
        loss_std_value_lst.append(loss_std_value)
    plt.figure()
    plt.plot(loss_mean_value_lst, label=f'Mean for Last {window}',marker='o')
    plt.xlabel('Harmful Corpus ID')
    plt.ylabel('Value')
    # pdb.set_trace()
    plt.legend()
    plt.tight_layout()
    # pdb.set_trace()
    # plt.savefig('test_mean.png')

    # plt.figure()
    # plt.plot(loss_std_value_lst, label=f'Std for Last {window}',marker='o')
    # plt.xlabel('Harmful Corpus ID')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('test_std.png')
    


    


    

# base_dir = "/scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_individual_corpus/attack_300_64_1.0_1"
# base_dir = "/scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_individual_corpus/attack_300_16_1.0_1"
# base_dir = "/scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_individual_corpus/attack_300_32_1.0_1"
# base_dir = "/scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/constrained/attack_individual_corpus/attack_300_64_1.0_1"
# base_dir = "/scratch/zx1673/codes/jailbreak_proj/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/results/attack_individual_corpus/attack_300_32_1.0_1"

# read_idv_attack_loss_file(base_dir)




