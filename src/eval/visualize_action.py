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

import h5py
import os
import cv2
import skimage
import matplotlib.pyplot as plt


def visualize_hand_pose(data, ax, offset=0):
    def remove_ticks(ax):
        # Hide grid lines
        ax.grid(False)
        ax.axis('off')
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    remove_ticks(ax)
    
    
    data = data + offset
    
    # Reorder them to make adjusting the plot view angle easier.
    plot_x = data[:, 2]
    plot_y = data[:, 0]
    plot_z = data[:, 1]
    linestyle = {'color':'red',
                'linewidth':3,
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
    
    ax.scatter3D(data[:,0], data[:,1], data[:,2], color='red', s=28)
    
    return ax

def visualize_joint_pose(data, ax, muscle=None):
    def remove_ticks(ax):
        # Hide grid lines
        ax.grid(False)
        ax.axis('off')
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.patch.set_alpha(0.0)
    
    remove_ticks(ax)

    # Reorder them to make adjusting the plot view angle easier.
    linestyle = {'color':'red',
                'linewidth':3,
                'markersize':12}
    
    # [4, 4, 7, 3, 4, 3, 3]
    
    # data[:, 0] *= -1.0
    # data[:, 1] *= -1.0
    # data[:, 2] *= -1.0
    
    
    leftleg = data[:4]
    for i in range(leftleg.shape[0]-1): 
        ax.plot([leftleg[i, 0], leftleg[i+1, 0]], [leftleg[i, 1] , leftleg[i+1, 1]],zs=[leftleg[i, 2],leftleg[i+1, 2]], **linestyle)
    
    rightleg = data[4: 8]
    for i in range(rightleg.shape[0]-1): 
        ax.plot([rightleg[i, 0], rightleg[i+1, 0]], [rightleg[i, 1] , rightleg[i+1, 1]],zs=[rightleg[i, 2],rightleg[i+1, 2]], **linestyle)
    
    spline = data[8: 16]
    for i in range(spline.shape[0]-1): 
        ax.plot([spline[i, 0], spline[i+1, 0]], [spline[i, 1] , spline[i+1, 1]],zs=[spline[i, 2],spline[i+1, 2]], **linestyle)
    
    hip = data[16:18]
    for i in range(hip.shape[0]-1): 
        ax.plot([hip[i, 0], hip[i+1, 0]], [hip[i, 1] , hip[i+1, 1]],zs=[hip[i, 2],hip[i+1, 2]], **linestyle)
    
    shoulders = data[20:23]
    for i in range(shoulders.shape[0]-1): 
        ax.plot([shoulders[i, 0], shoulders[i+1, 0]], [shoulders[i, 1] , shoulders[i+1, 1]],zs=[shoulders[i, 2],shoulders[i+1, 2]], **linestyle)
        
    # leftarm = np.concatenate([data[22:21], data[23:25]], axis=0)
    leftarm = data[22:25]
    for i in range(leftarm.shape[0]-1): 
        ax.plot([leftarm[i, 0], leftarm[i+1, 0]], [leftarm[i, 1] , leftarm[i+1, 1]],zs=[leftarm[i, 2],leftarm[i+1, 2]], **linestyle)
    
    shoulders = np.concatenate([data[21:22], data[26:27]], axis=0)
    for i in range(shoulders.shape[0]-1): 
        ax.plot([shoulders[i, 0], shoulders[i+1, 0]], [shoulders[i, 1] , shoulders[i+1, 1]],zs=[shoulders[i, 2],shoulders[i+1, 2]], **linestyle)
        
    rightarm = data[26:]
    for i in range(rightarm.shape[0]-1): 
        ax.plot([rightarm[i, 0], rightarm[i+1, 0]], [rightarm[i, 1] , rightarm[i+1, 1]],zs=[rightarm[i, 2],rightarm[i+1, 2]], **linestyle)
    
    ax.scatter3D(data[:,0], data[:,1], data[:,2], color='red', s=28)
    
    ax.set_xlim3d(0.0, 1.0)
    ax.set_ylim3d(-0.1, 1.1)
    ax.set_zlim3d(data[:,2].min(), data[:, 2].max())
    
    
    if muscle is not None:
        coord = np.linspace(data[:,2].min(), data[:, 2].max(), 8)
        ax.scatter3D( 0.0, 0.0, coord, c=muscle[0], s=30, cmap="Greys", vmin=0.0, vmax=1.0)
        ax.scatter3D( 0.0, 1.0, coord, c=muscle[1], s=30, cmap="Greys", vmin=0.0, vmax=1.0)
    
    return ax   
    
def visualize_emg(data, ax):
    def remove_ticks(ax):
        # Hide grid lines
        ax.grid(False)
        ax.axis('off')
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
    remove_ticks(ax)
    
    contour = np.load(f"emg_contour.npy")
    im = ax.imshow(contour, cmap="Greys", vmin=0.0, vmax=1.0)   
    
    x_range = [60, 650] 
    y_range = [45, 535]
    
    # map data coordinates to hand plot coordinate
    y_coord = np.linspace(*y_range, 8)
    x_coord = data * (600) + 60
    
    ax.scatter( x_coord, y_coord,  c=data, s=28, cmap="Greys", vmin=0.0, vmax=1.0)

    return ax
     
def visualize_tactile(data, ax, side='left'):
    def remove_ticks(ax):
        # Hide grid lines
        ax.grid(False)
        ax.axis('off')
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
    remove_ticks(ax)
    
    data= data[0]
    mapping = np.load(f"{side}_mapping.npy")
    contour = np.load(f"{side}.npy")
    im = ax.imshow(contour, cmap="Greys", vmin=0.0, vmax=1.0)    
    
    # map data coordinates to hand plot coordinate
    idx_coords = mapping[:, :2].astype(np.int16)
    c = data[idx_coords[:,0], idx_coords[:,1]] 
    x = mapping[:,2]
    y = mapping[:,3]
    # self._ax.scatter(x, y, s=15, c=c, cmap="inferno") #, vmin=0, vmax=1
    ax.scatter(x, y, s=15, c=c, cmap="Greys", vmin=0.0, vmax=1.0)

    return ax
    
    
if __name__ == '__main__':
    signals = ['myo-emg-left', 'myo-emg-right', 'tactile-glove-left', 'tactile-glove-right', 'right-hand-pose', 'left-hand-pose', 'joint-position']
    data_resample_len = 12
    plot_r = 12
    exp_suffix = 'test_feature_individual_activation'
    batch_size = 8
    num_workers = 10
    eval_freq = 20
    device = 'cuda:5'
    save_folder = 'action_visual_hist'

    # train_dataset = ActionSenseVideo(path='./Dataset/', split='train', parse_signal_keys=signals, resample_len=data_resample_len)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset = ActionSenseVideo(path='./Dataset/', split='val', parse_signal_keys=signals, resample_len=data_resample_len, video_res=64)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)


    offsets = [np.array([0.0, 0.2, 0.0]), np.array([0.5, 0.0, 0.0])]
    # train for every epoch
    for val_ind, val_batch in enumerate(tqdm(val_dataloader)):
        # if val_ind < 1:
        #     continue
        fig = plt.figure(figsize=(55, 10)) 
        for time_step in range(val_batch['signal'][signals[0]].shape[1]):
            for sig in signals:
                if 'left-hand-pose' in sig:
                    ax = fig.add_subplot(5, plot_r*2, 2*time_step+1,  projection='3d') 
                    visualize_hand_pose(val_batch['signal'][sig][0, time_step], ax) #  offsets[signals.index(sig)%2]
                if 'tactile-glove-left' in sig:
                    ax = fig.add_subplot(5, plot_r*2, 2*time_step+2) 
                    visualize_tactile(val_batch['signal'][sig][0, time_step], ax, side='left')
                if 'right-hand-pose' in sig:
                    ax = fig.add_subplot(5, plot_r*2, plot_r*2 + 2*time_step+1 , projection='3d') 
                    visualize_hand_pose(val_batch['signal'][sig][0, time_step], ax) #  offsets[signals.index(sig)%2]
                if 'tactile-glove-right' in sig:
                    ax = fig.add_subplot(5, plot_r*2, plot_r*2 + 2*time_step+2) 
                    visualize_tactile(val_batch['signal'][sig][0, time_step], ax, side='right')
                if 'joint-position' in sig:
                    ax = fig.add_subplot(3, plot_r, plot_r+time_step+1,  projection='3d') 
                    visualize_joint_pose(val_batch['signal'][sig][0, time_step], ax, muscle=None)
                if 'emg-left' in sig:
                    ax = fig.add_subplot(5, plot_r*2, 3*(plot_r*2) + 2*time_step+1) 
                    visualize_emg(val_batch['signal'][sig][0, time_step].numpy(), ax)
                if 'emg-right' in sig:
                    ax = fig.add_subplot(5, plot_r*2, 3*(plot_r*2) + 2*time_step+2) 
                    visualize_emg(val_batch['signal'][sig][0, time_step].numpy(), ax)
        
        text = val_batch['label_text'][0][0]
        fig.savefig(f'./{save_folder}/action_{val_ind}_{text}.png')
        plt.close(fig)
