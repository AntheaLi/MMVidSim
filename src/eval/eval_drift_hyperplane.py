from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import cv2
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from subprocess import call
import importlib

from torchvision import transforms
import imageio
import numpy as np
from os.path import splitext
import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import numpy as np

# import tensorflow as tf
from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
from einops import rearrange
from torchvision import transforms as T, utils
import imageio

exp_name = sys.argv[1] if len(sys.argv) >= 2 else 'gem' 


def compute_optical_flow(prev_frame, next_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def compute_temporal_drift_error(predicted_video, ground_truth_video):
    num_frames = predicted_video.shape[0]
    temporal_drift_error = 0.0
    
    for i in range(num_frames - 1):
        # Compute optical flow for consecutive frames
        pred_flow = compute_optical_flow(predicted_video[i], predicted_video[i + 1])
        gt_flow = compute_optical_flow(ground_truth_video[i], ground_truth_video[i + 1])
        
        # Compute the difference between the predicted and ground truth optical flows
        flow_diff = np.linalg.norm(pred_flow - gt_flow, axis=2)
        
        # Sum the differences to get the temporal drift for this pair of frames
        temporal_drift_error += np.sum(flow_diff)
    
    # Normalize by the number of frames
    temporal_drift_error /= (num_frames - 1)
    return temporal_drift_error

if __name__ == '__main__':
    exp_dir = os.path.join('trained_ckpt', exp_name)
    results_dir = glob(exp_dir + f'/visu_*')[0]
    print(results_dir)
    all_results = glob(results_dir+'/metric/*.npy')
    all_results = sorted(all_results)
    os.makedirs(results_dir + '/visu', exist_ok=True)
    os.makedirs('./all/drift', exist_ok=True)
    all_drift = []
    
    f = open(results_dir + '/metric/metrics.txt', 'w')
    
    B = np.load(all_results[0]).shape[0]
    
    gt_indices = np.arange(B).reshape(-1, 2)[:, 0]
    pred_indices = np.arange(B).reshape(-1, 2)[:, 1]
    
    total_videos = B * len(all_results)
    total_round = total_videos // 16
    
    for i in range(len(all_results)):
        current_results = np.load(all_results[i])
        if current_results.shape[0] < B:
            continue
        current_results = current_results[:B] 
        
        # B, F, C, H, W = current_results.shape
        gt_results = np.array(current_results[gt_indices].transpose(0, 1, 3, 4, 2) * 255, dtype=np.uint8)
        pred_results = np.array(current_results[pred_indices].transpose(0, 1, 3, 4, 2) * 255, dtype=np.uint8)
        
        current_batch_drift = []
        for j in range(B//2):
            current_batch_drift.append(compute_temporal_drift_error(pred_results[j], gt_results[j]))
            imageio.mimsave(f'{results_dir}/visu/pred_{i*B+j}.gif', pred_results[j],fps=10)
            imageio.mimsave(f'{results_dir}/visu/gt_{i*B+j}.gif', gt_results[j],fps=10)
            
        print(i, np.mean(current_batch_drift))
            
        all_drift.extend(current_batch_drift)
    
    print(exp_name, ':')
    print(f'Drift: {np.mean(all_drift)}')
    with open(f'./all/drift/{exp_name}.txt', 'w') as f:
        f.write(f'Drift: {np.mean(all_drift)}')
    


