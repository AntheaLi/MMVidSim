import os
import time
import sys
sys.path.append('./')
sys.path.append('../')
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
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from subprocess import call
import importlib

from torchvision import transforms
import imageio
import torch
from os.path import splitext
from data import ActionSenseVideo
from models.model_hyperplane_trainer import Trainer
from models.model_video_hyperplane import VideoDiffusion, Network
# from exp.action.models.encoders import TextEncoder
from models.encoders import TextEncoder, ImageEncoder
from models.model_modules import ResNetImageEncoder

from exp.utils import dict_to_device
from einops import rearrange
from torchvision import transforms as T, utils


# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 


if __name__ == '__main__':

    basedir = 'trained_ckpt'
    ckpt = 'open-clip-res-train-global-mlp'
    exp_note = ''
    ckpt_number = 2 # 6, 12, 15, 16
    exp_dir = os.path.join(basedir, ckpt)
    checkpoint = os.path.join(exp_dir, f'model-{ckpt_number}.pt')
    args = torch.load(os.path.join(exp_dir, 'ckpts', 'args.pt'))
    args.data_path = './Dataset_local'
    visu_dir = os.path.join(exp_dir, f'visu_{ckpt_number}_{exp_note}')
    
    signals = ['myo-emg-left', 'myo-emg-right', 'tactile-glove-left', 'tactile-glove-right', 'right-hand-pose', 'left-hand-pose', 'joint-position']

    train_dataset = ActionSenseVideo(path=args.data_path, split='train', parse_signal_keys=signals, resample_len=args.data_resample_len, video_res=args.image_size)

    val_dataset = ActionSenseVideo(path=args.data_path, split='val', parse_signal_keys=signals, resample_len=args.data_resample_len, video_res=args.image_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # experiment logging and directory:
    device = 'cuda:0'
    # create models
    args.signal_model_config = parse_signal_model_config(signals, args)
    model = Network(args, cond_drop_chance=args.cond_drop_chance, signals=signals)
    if args.pretrain != '': model._load_pretrained_signal_encoder(args.pretrain)
    text_model = TextEncoder(model_type=args.text_model)
    text_model = text_model.to(device)
    text_model.requires_grad_(False)
    text_model.eval()
    
    image_model = None
    if 'metric' in args.hyperplane: 
        # image_model = ResNetImageEncoder()
        image_model = ImageEncoder(model_type=args.image_model)
        image_model = image_model.to(device)
        image_model.requires_grad_(False)
        image_model.eval()
        
    
    trainer = Trainer(diffusion_model=model,
                      model_type=args.video_model,
                      condition_encoder=text_model, 
                      context_encoder=image_model, 
                      train_set=train_dataset,
                      valid_set=val_dataset,
                      results_folder=exp_dir,
                      use_text_condition=False,
                      cond_drop_chance=args.cond_drop_chance,
                      train_batch_size=args.batch_size,
                      valid_batch_size=args.batch_size // 2, 
                      guide_scale=args.guide_scale,
                      **args.train_config)


    # create logs
    header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR     Loss'
        
    trainer.load(checkpoint)

    batch_size = 5
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)
    
    pbar = tqdm(val_loader, leave=False)
    for eval_batch_ind, batch in enumerate(pbar):
        batch = dict_to_device(batch)
        gt_video = batch['video'].permute(0, 4, 1, 2, 3)
        b, c, t, h, w = gt_video.shape
        image = gt_video[:, :, 0:1]
        output = trainer.sample(image, batch, batch_size=batch_size)
        output_w_first_frame = torch.cat([image, output], dim=2)
        assert output_w_first_frame.shape[2] == t
        gt_video = rearrange(gt_video, 'b c t h w -> b t c h w')
        output_w_first_frame = rearrange(output_w_first_frame, 'b c t h w -> b t c h w')
        
        
        to_plot = []
        # (2, 3, 8, 64, 64)
        to_plot.append(gt_video[0].reshape(-1, 3, 64, 64))
        to_plot.append(output_w_first_frame[0].reshape(-1, 3, 64, 64))
        to_plot.append(gt_video[1].reshape(-1, 3, 64, 64))
        to_plot.append(output_w_first_frame[1].reshape(-1, 3, 64, 64))
        to_plot.append(gt_video[2].reshape(-1, 3, 64, 64))
        to_plot.append(output_w_first_frame[0].reshape(-1, 3, 64, 64))
        to_plot.append(gt_video[3].reshape(-1, 3, 64, 64))
        to_plot.append(output_w_first_frame[1].reshape(-1, 3, 64, 64))
        to_plot.append(gt_video[4].reshape(-1, 3, 64, 64))
        to_plot.append(output_w_first_frame[1].reshape(-1, 3, 64, 64))
        
        plot = torch.stack(to_plot)
        plot = rearrange(plot, 'b n c h w -> (b n) c h w', n=t)
        output_gif = os.path.join(visu_dir, f'out_{eval_batch_ind}.png')
        utils.save_image(plot, output_gif, nrow=t)
        print(f'Generated {output_gif}')








