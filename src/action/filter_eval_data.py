import os, sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from einops import einsum, repeat
from tqdm import tqdm 

# from config import text_model_cfg, signal_model_cfg, projection_config, loss_weight_config
from utils import *
from tqdm import tqdm
from data import ActionSenseVideo
# from config import BaseArgs, parse_signal_model_config
from tensorboardX import SummaryWriter
from time import strftime, gmtime, time


if __name__ == '__main__':
    signals = ['myo-emg-left', 'myo-emg-right', 'tactile-glove-left', 'tactile-glove-right', 'right-hand-pose', 'left-hand-pose', 'joint-position']
    data_resample_len = 9
    exp_suffix = 'test_feature_individual_activation'
    batch_size = 8
    num_workers = 10
    eval_freq = 20
    device = 'cuda:5'

    # train_dataset = ActionSenseVideo(path='./Dataset/', split='train', parse_signal_keys=signals, resample_len=data_resample_len)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset = ActionSenseVideo(path='./Dataset/', split='val', parse_signal_keys=signals, resample_len=data_resample_len, video_res=64)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)


    offsets = [np.array([0.0, 0.2, 0.0]), np.array([0.5, 0.0, 0.0])]
    # train for every epoch
    for val_ind, val_batch in enumerate(tqdm(val_dataloader)):
        # if val_ind < 1:
        #     continue
        fig = plt.figure(figsize=(50, 10)) 
        for eval_batch_ind, batch in enumerate(pbar):
            batch = dict_to_device(batch)
            gt_video = batch['video'].permute(0, 4, 1, 2, 3)
            b, c, t, h, w = gt_video.shape
        fig.savefig(f'./action_visual/action_{val_ind}.png')
        plt.close(fig)