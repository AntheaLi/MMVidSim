import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from torchvision import transforms as T, utils


base_dir = '/data/vision/torralba/scratch/yichenl/projects/source/act/exp/eval/trained_ckpt/'
expn = sys.argv[1] if len(sys.argv) > 1 else 'ours-4'
visu = sys.argv[2] if len(sys.argv) > 2 else 'visu22_v100'

os.makedirs(f'vis/{expn}/{visu}', exist_ok=True)
to_vis = sorted(glob(f'{base_dir}/{expn}/{visu}/metric/*.npy'))
counter = 0

b, n, c, h, w = np.load(to_vis[0]).shape # b n c h w

for fname in tqdm(to_vis):
    plot = np.load(fname) # b n c h w
    for i in range(b//2): 
        cur_plot = plot[2*i:2*i+2].reshape(-1, c, h, w)
        output_gif = os.path.join(f'vis/{expn}/{visu}', f'out_{counter}.png')
        utils.save_image(torch.from_numpy(cur_plot), output_gif, nrow=n)
        counter += 1
