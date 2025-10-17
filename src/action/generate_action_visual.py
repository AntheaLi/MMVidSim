import os, sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from einops import einsum, repeat

from config import text_model_cfg, signal_model_cfg, projection_config, loss_weight_config
from utils import *
from tqdm import tqdm
from data import ActionSenseVideo
from config import BaseArgs, parse_signal_model_config
from tensorboardX import SummaryWriter
from time import strftime, gmtime, time

import h5py
import os
import cv2
import skimage
import matplotlib.pyplot as plt



def visualize_hand_pose(data, ax):
    def remove_ticks(ax):
        # Hide grid lines
        ax.grid(False)
        ax.axis('off')
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    remove_ticks(ax)
    
    
    # Reorder them to make adjusting the plot view angle easier.
    plot_x = data[:, 2]
    plot_y = data[:, 0]
    plot_z = data[:, 1]
    linestyle = {'color':'red',
                'linewidth':2,
                'markersize':12}
    
    thumb = data[:4]
    for i in range(thumb.shape[0]-1): 
        ax.plot([thumb[i, 0], thumb[i+1, 0]], [thumb[i, 1] , thumb[i+1, 1]],zs=[thumb[i, 2],thumb[i+1, 2]], **linestyle)
    
    index = data[4: 9]
    for i in range(index.shape[0]-1): 
        ax.plot([index[i, 0], index[i+1, 0]], [index[i, 1] , index[i+1, 1]],zs=[index[i, 2],index[i+1, 2]], **linestyle)
    
    middle = data[9 : 9+5]
    for i in range(middle.shape[0]-1): 
        ax.plot([middle[i, 0], middle[i+1, 0]], [middle[i, 1] , middle[i+1, 1]],zs=[middle[i, 2],middle[i+1, 2]], **linestyle)
    
    ring = data[9+5:19]
    for i in range(ring.shape[0]-1): 
        ax.plot([ring[i, 0], ring[i+1, 0]], [ring[i, 1] , ring[i+1, 1]],zs=[ring[i, 2],ring[i+1, 2]], **linestyle)
    
    
    pinky = data[19:]
    for i in range(pinky.shape[0]-1): 
        ax.plot([pinky[i, 0], pinky[i+1, 0]], [pinky[i, 1] , pinky[i+1, 1]],zs=[pinky[i, 2],pinky[i+1, 2]], **linestyle)
    
    ax.scatter3D(data[:,0], data[:,1], data[:,2], )
    
    return ax



if __name__ == '__main__':
        
    signals = ['myo-emg-left', 'myo-emg-right', 'tactile-glove-left', 'tactile-glove-right', 'right-hand-pose', 'left-hand-pose', 'joint-position']
    data_resample_len = 9
    exp_suffix = 'test_feature_individual_activation'
    batch_size = 8
    num_workers = 10
    eval_freq = 20
    device = 'cuda:5'

    train_dataset = ActionSenseVideo(path='./Dataset/', split='train', parse_signal_keys=signals, resample_len=data_resample_len)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset = ActionSenseVideo(path='./Dataset/', split='val', parse_signal_keys=signals, resample_len=data_resample_len)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)



    # train for every epoch
    for val_ind, val_batch in enumerate(val_dataloader):
        if val_ind < 10:
            continue
        for sig in signals:
            if 'hand-pose' in sig:
                # visualize hand pose data
                fig, axs = plt.subplots(nrows=1, ncols=9,
                                squeeze=False, # if False, always return 2D array of axes
                                # sharex=True, sharey=True,
                                subplot_kw={'projection': '3d'},
                                figsize=(40, 5)
                                )
                for time_step in range(val_batch['signal'][sig].shape[1]):
                    print(val_batch['signal'][sig][0, time_step])
                    axs[0][time_step] = visualize_hand_pose(val_batch['signal'][sig][0, time_step], axs[0][time_step])
                plt.show()
