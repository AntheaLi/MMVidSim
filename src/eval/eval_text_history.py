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


from data import ActionSenseVideo
from models.model_hyperplane_history_trainer import Trainer
from models.model_video_history_hyperplane import VideoDiffusion, Network
# from exp.action.models.encoders import TextEncoder
from models.encoders import TextEncoder, ImageEncoder
from models.model_modules import ResNetImageEncoder

from exp.utils import dict_to_device
from einops import rearrange
from torchvision import transforms as T, utils


# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 


if __name__ == '__main__':   
    ckpt = sys.argv[1]  if len(sys.argv) > 1 else 'text-4'
    ckpt_number = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sampling_timestep = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    exp_note = sys.argv[4] if len(sys.argv) > 4 else 'eval'
    
    print(f'\n\n evaluating data from {ckpt} at checkpoint {ckpt_number} at sampling timestep {sampling_timestep} ... \n\n' )
    
    basedir = 'trained_ckpt'
    exp_dir = os.path.join(basedir, ckpt)
    checkpoint = os.path.join(exp_dir, f'model-{ckpt_number}.pt')
    visu_dir = os.path.join(exp_dir, f'visu_{ckpt_number}_sampling{sampling_timestep}_{exp_note}')
    os.makedirs(visu_dir, exist_ok=True)
    metric_dir = os.path.join(exp_dir, f'visu_{ckpt_number}_sampling{sampling_timestep}_{exp_note}', 'metric')
    os.makedirs(metric_dir, exist_ok=True)
    args = torch.load(os.path.join(exp_dir, 'ckpts', 'args.pt'))
    args.data_path = './Dataset/'
    
    
    signals = ['myo-emg-left', 'myo-emg-right', 'tactile-glove-left', 'tactile-glove-right', 'right-hand-pose', 'left-hand-pose', 'joint-position']


    train_dataset = ActionSenseVideo(path=args.data_path, split='train', parse_signal_keys=signals, horizon=4, resample_len=args.data_resample_len, video_res=args.image_size)
    val_dataset = ActionSenseVideo(path=args.data_path, split='val', parse_signal_keys=signals, horizon=4, resample_len=args.data_resample_len, video_res=args.image_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # experiment logging and directory:
    device = 'cuda:0'
    # create models
    args.signal_model_config = parse_signal_model_config(signals, args)
    importlib.invalidate_caches()
    text_model = TextEncoder(model_type=args.text_model)
    text_model.requires_grad_(False)
    text_model.eval()

    diffusion = VideoDiffusion(args, use_text=True, model_type=args.video_model)
    
    trainer = Trainer(diffusion_model=diffusion,
                      model_type=args.video_model,
                      condition_encoder=text_model, 
                      context_encoder=None,
                      train_set=train_dataset,
                      valid_set=val_dataset,
                      results_folder=exp_dir,
                      use_text_condition=True,
                      train_batch_size=args.batch_size,
                      valid_batch_size=args.eval_batch_size, 
                      guide_scale=args.guide_scale,
                      **args.train_config)



    # create logs
    header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR     Loss'
        
    trainer.load(checkpoint)

    batch_size = 12
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=32)
    
    pbar = tqdm(val_loader, leave=False)
    total_mean_error = []
    total_frame_acc_error = [] 
    for eval_batch_ind, batch in enumerate(pbar):
        batch = dict_to_device(batch)
        gt_video = batch['video'].permute(0, 4, 1, 2, 3)
        b, c, t, h, w = gt_video.shape
        image = gt_video[:, :, 0:4]
        output = trainer.sample(image, batch, batch_size=batch_size)
        output_w_first_frame = torch.cat([image, output], dim=2)
        assert output_w_first_frame.shape[2] == t
        gt_video = rearrange(gt_video, 'b c t h w -> b t c h w')
        output_w_first_frame = rearrange(output_w_first_frame, 'b c t h w -> b t c h w')
        
        error_metric = torch.sqrt((gt_video[:, 1:] - output_w_first_frame[:, 1:])**2)
        total_mean_error.append(error_metric.detach().cpu().numpy().mean())
        total_frame_acc_error.append(error_metric.sum(1).detach().cpu().numpy().mean())
        
        to_plot = []
        # (2, 3, 8, 64, 64)
        for i in range(batch_size):
            to_plot.append(gt_video[i].reshape(-1, 3, 64, 64))
            to_plot.append(output_w_first_frame[i].reshape(-1, 3, 64, 64))
        
        plot = torch.stack(to_plot)
        output_npy = os.path.join(metric_dir, f'batch_{eval_batch_ind}_{error_metric.mean()}.npy')
        np.save(output_npy, plot.clone().detach().cpu().numpy())
        plot = rearrange(plot, 'b n c h w -> (b n) c h w', n=t)
        output_gif = os.path.join(visu_dir, f'out_{eval_batch_ind}.png')
        utils.save_image(plot, output_gif, nrow=t)
        print(f'Generated {output_gif}, error batch {error_metric.mean()}')

    total_mean_error = np.mean(total_mean_error)
    total_frame_acc_error = np.mean(total_frame_acc_error)
    print(f'average mean error = {total_mean_error}')
    print(f'average frame error = {total_frame_acc_error}')
    with open(os.path.join(visu_dir, 'eval_metric.txt'), 'w') as f:
        f.write(f'average mean error = {total_mean_error} \n')
        f.write(f'average frame error = {total_frame_acc_error}')
        
        






