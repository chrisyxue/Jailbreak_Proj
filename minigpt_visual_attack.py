# 1. Download Vicuna's weights to ./models   (it's a delta version)
# 2. Download LLaMA's weight via: https://huggingface.co/huggyllama/llama-13b/tree/main
# 3. merge them and setup config
# 4. Download the mini-gpt4 compoents' pretrained ckpts
# 5. vision part will be automatically download when launching the model


import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from minigpt_utils import visual_attacker, prompt_wrapper

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from harmful_corpus.data_info import data_info_dict
import pandas as pd
import csv

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/zx1673/codes/jailbreak_proj/model"


from utils import render_save_dir
import pdb


VICUNA_7b_PATH = '/scratch/zx1673/codes/jailbreak_proj/model/vicuna-7b'
MINIGPT_VICUNA_7b_PATH = '/scratch/zx1673/codes/jailbreak_proj/model/mini-gpt4/prerained_minigpt4_7b.pth'

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.",choices=['eval_configs/minigpt4_eval.yaml', 'eval_configs/minigpt4_eval_llama.yaml', 'eval_configs/minigpt4_eval_vicuna7b.yaml'])
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size duing attacking")
    parser.add_argument('--alpha', type=float, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument('--seed',default=1024,type=int,help="random seed for reproducibility")
    
    # llm model
    parser.add_argument("--llm_model", type=str, default="vicuna", help="the model of the llm", choices=['vicuna', 'vicuna_7b'])
    
    # Parameter for Individual Corpus
    parser.add_argument("--attack_individual_corpus", default=False, action='store_true')
    parser.add_argument("--start_corpus", default=0, type=int, help='start index of the corpus')

    # Paramter for Subset of Corpus
    parser.add_argument("--subset_corpus_num", default=-1, type=int, help='number of corpus to attack (-1) means all the corpus')

    # Paramter for Corpus Selection
    parser.add_argument("--corpus_selection_num", default=-1, type=int, help='number of corpus to attack (-1) means select all the corpus')
    parser.add_argument('--select_method', type=str, default='Detoxify_Rank_Discreate', choices=['Detoxify_Rank_Discreate','Detoxify_Rank_Continuous', 'Perspective_Rank_Discreate', 'Perspective_Rank_Continuous'],help='method to select the corpus')
    parser.add_argument('--rever_select',  default=False, action='store_true', help='select the corpus reversely')

    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")
    parser.add_argument("--debug", default=False, action='store_true', help='debug mode')

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    if args.attack_individual_corpus:
        args.batch_size = 1

    args.save_dir = render_save_dir(args)
    return args


def setup_seeds(config,seed):
    # seed = config.run_cfg.seed + get_rank()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)
setup_seeds(cfg,args.seed)
# TODO: replace some parameters in cfg based on args
# Edit Config based in Model Arch
if args.llm_model == 'vicuna':
    pass
elif args.llm_model == 'vicuna_7b':
    cfg.model_cfg.llama_model= VICUNA_7b_PATH
    cfg.model_cfg.ckpt = MINIGPT_VICUNA_7b_PATH


model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
if args.debug:
    pdb.set_trace()
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))


model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]\n')


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)


# Load Corpus
file = open("harmful_corpus/derogatory_corpus.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
for i in range(num):
    targets.append(data[i][0])

# file_detoxify = pd.read_csv("harmful_corpus/detoxify_detoxify_res.csv")

if args.attack_individual_corpus:
    origi_save_dir = args.save_dir
    target_id_lst = [i for i in range(args.start_corpus, len(targets))]
    for target_id in target_id_lst:
        args.save_dir = os.path.join(origi_save_dir, f'corpus_{target_id}')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        print('Save Dir: ' + args.save_dir)
        target = [targets[target_id]]
        
        my_attacker = visual_attacker.Attacker(args, model, target, device=model.device, is_rtp=False)
        template_img = 'adversarial_images/clean.jpeg'
        img = Image.open(template_img).convert('RGB')
        img = vis_processor(img).unsqueeze(0).to(model.device)

        text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_no_text_input

        if not args.constrained:
            adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                                img=img, batch_size = args.batch_size,
                                                                num_iter=args.n_iters, alpha=args.alpha/255)

        else:
            adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
                                                                img=img, batch_size= args.batch_size,
                                                                num_iter=args.n_iters, alpha=args.alpha / 255,
                                                                epsilon=args.eps / 255)
            
        
        bmp_image_path = f'{args.save_dir}/bad_prompt.bmp'
        save_image(adv_img_prompt, bmp_image_path)
        with Image.open(bmp_image_path) as img:
            jpg_image_path = bmp_image_path.replace('.bmp', '.jpg')
            img.save(jpg_image_path, 'JPEG')      

else:
    if args.subset_corpus_num > 0:
        targets = targets[:args.subset_corpus_num]
    if args.corpus_selection_num > 0:
        if args.rever_select:
            selec_idx = data_info_dict['derogatory_corpus'][args.select_method][-args.corpus_selection_num:]
        else:
            selec_idx = data_info_dict['derogatory_corpus'][args.select_method][:args.corpus_selection_num]
        targets = [targets[i] for i in selec_idx]

        # pdb.set_trace()    
    my_attacker = visual_attacker.Attacker(args, model, targets, device=model.device, is_rtp=False)
    template_img = 'adversarial_images/clean.jpeg'
    img = Image.open(template_img).convert('RGB')
    img = vis_processor(img).unsqueeze(0).to(model.device)

    text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_no_text_input



    if not args.constrained:
        adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                                img=img, batch_size = args.batch_size,
                                                                num_iter=args.n_iters, alpha=args.alpha/255)

    else:
        adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
                                                                img=img, batch_size= args.batch_size,
                                                                num_iter=args.n_iters, alpha=args.alpha / 255,
                                                                epsilon=args.eps / 255)

    save_image(adv_img_prompt, '%s/bad_prompt_test.bmp' % args.save_dir)
    print('[Done]')
