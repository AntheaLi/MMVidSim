import os
import time
import sys
import shutil
import random
from copy import deepcopy
from tqdm import tqdm
from time import strftime
from datetime import datetime
from itertools import chain
from config import parse_args, parse_signal_model_config

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from subprocess import call
import importlib

from data import ActionSenseVideoVerb
# from models.model_wrapper import Trainer
from models.model_hyperplane_trainer import Trainer
# from models.model_video import VideoDiffusion
from models.model_video_hyperplane import VideoDiffusion
from models.encoders import TextEncoder, SignalEncoder

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 


if __name__ == '__main__':
    args = parse_args()

    signals = ['myo-emg-left', 'myo-emg-right', 'tactile-glove-left', 'tactile-glove-right', 'right-hand-pose', 'left-hand-pose', 'joint-position']

    train_dataset = ActionSenseVideoVerb(path=args.data_path, split='train', parse_signal_keys=signals, resample_len=args.data_resample_len, video_res=args.image_size)

    val_dataset = ActionSenseVideoVerb(path=args.data_path, split='val', parse_signal_keys=signals, resample_len=args.data_resample_len, video_res=args.image_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


   # experiment logging and directory:
    start_time = datetime.now()
    exp_dir = f"./{args.log_dir}/{args.exp_suffix}-{str(start_time).replace(' ', '-')}/"
    best_ckpt = f'./{exp_dir}/ckpts/best_ckpt.pth'
    device = 'cuda:0'
    os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, 'ckpts'))
    os.makedirs(os.path.join(exp_dir, 'visu'))
    # os.system(f'cp data.py models/{args.model}.py {__file__} {exp_dir}')

    # create models
    args.signal_model_config = parse_signal_model_config(signals, args)
    importlib.invalidate_caches()
    text_model = TextEncoder(model_type=args.text_model)
    text_model.requires_grad_(False)
    text_model.eval()

    
    torch.save(args, f'{exp_dir}/ckpts/args.pt')

    diffusion = VideoDiffusion(args, use_text=True, model_type=args.video_model)
    
    trainer = Trainer(diffusion_model=diffusion,
                      model_type=args.video_model,
                      condition_encoder=text_model, 
                      context_encoder=None,
                      train_set=train_dataset,
                      valid_set=val_dataset,
                      results_folder=exp_dir,
                      use_text_condition=True,
                      cond_drop_chance=args.cond_drop_chance,
                      train_batch_size=args.batch_size,
                      valid_batch_size=args.batch_size // 2, 
                      guide_scale=args.guide_scale,
                      **args.train_config)


    # create logs
    header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR     Loss'

    if args.ckpt is not None:
        trainer.load(args.ckpt)
        print('loaded', args.ckpt)
    
    trainer.train()
