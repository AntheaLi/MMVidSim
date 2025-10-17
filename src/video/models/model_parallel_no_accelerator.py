

import os
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
import torch.utils.checkpoint
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
import numpy as np
from pynvml import *

from exp.utils import dict_to_device


def cycle(dl):
    while True:
        for data in dl:
            yield data

def exists(x):
    return x is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# trainer class

class Trainer(object):
    def __init__(
        self,
        args,
        diffusion_model,
        condition_encoder, 
        train_dataloader,
        valid_set,
        sampler=None,
        channels = 3,
        *,
        gpu='cuda:0',
        valid_batch_size = 1,
        main_process=False, 
        model_type = 'avdc',
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 1,
        guide_scale=0.0,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048, 
        cond_drop_chance=0.1,
        use_text_condition=True
    ):
        super().__init__()

        assert model_type in ['avdc', 'i2vgen', 'i2vgen-avdc', 'our', 'ours'], f'{model_type} not implemented'

        self.model_type = model_type

        self.cond_drop_chance = cond_drop_chance

        self.condition_encoder = condition_encoder

        self.use_text_condition = use_text_condition

        self.guide_scale = guide_scale

        self.gpu = gpu

        self.args = args
        
        self.main_process = main_process


        if condition_encoder is None:
            assert not self.use_text_condition, f' not conditioning model passed in'
        # accelerator

        self.model = diffusion_model

        self.channels = channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.module.image_size

        # dataset and dataloader
        valid_ind = [i for i in range(len(valid_set))][:num_samples]

        valid_set = Subset(valid_set, valid_ind)

        self.valid_ds = valid_set
        dl = train_dataloader
        self.dl = cycle(dl)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        # step counter state

        self.step = 0


    @property
    def device(self):
        return self.gpu

    def save(self, milestone):
        if not self.main_process:
            return

        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            # 'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        device = self.gpu

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        # model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        # if self.accelerator.is_main_process:
        if self.main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        # if exists(self.accelerator.scaler) and exists(data['scaler']):
        #     self.accelerator.scaler.load_state_dict(data['scaler'])


    def sample(self, batch, batch_condition, batch_size=1, guide_scale=0):
        device = self.device
        task_embeds = self.encode_action(batch_condition)
        return self.ema.ema_model.module.sample(image=batch.to(device), action_data=task_embeds.to(device), batch_size=batch_size, guide_scale=guide_scale)


    def encode_action(self, batch):
        if self.use_text_condition: 
            goal = batch['label_text']
            goal_embed = self.condition_encoder(goal)
            goal_embed = goal_embed * (torch.rand(goal_embed.shape[0], 1, 1, device = goal_embed.device) > self.cond_drop_chance).float()

        else: 
            goal_embed = batch['signal']


        if self.model_type == 'i2vgen' and self.guide_scale > 0:
            negative_goal_emb = self.condition_encoder(batch['negative_label_text'])
            if negative_goal_emb.ndim == 2: negative_goal_emb = negative_goal_emb.unsqueeze(1)
            return goal_embed, negative_goal_emb
            
        
        return goal_embed


    def eval(self):

        # accelerator = self.accelerator
        # device = accelerator.device
        device = self.gpu
        self.ema.ema_model.eval()

        with torch.no_grad():
            milestone = self.step // self.save_and_sample_every
            batches = num_to_groups(self.num_samples, self.valid_batch_size)
            ### get val_imgs from self.valid_dl
            x_conds = []
            xs = []
            task_embeds = []
            for i, batch in enumerate(self.valid_dl):
                batch = dict_to_device(batch, device)
                val_goal_emb = self.encode_action(batch)
                task_embeds.append(val_goal_emb)

                if self.model_type == 'avdc':
                    x = batch['video'].permute(0, 1, 4, 2, 3)
                    b, t, c, h, w = x.shape
                    x_cond = x[:, 0:1]
                    x_conds.append(x_cond.reshape(b, 1 * c, h, w))
                    xs.append(x[:, 1:].reshape(b, (t - 1) * c, h, w))

                elif 'i2vgen' in self.model_type or 'our' in self.model_type:

                    x = batch['video'].permute(0, 4, 1, 2, 3)
                    b, c, t, h, w = x.shape
                    x_conds.append(x[:, :, 0:1])
                    xs.append(x[:, :, 1:])

                else:
                    raise NotImplementedError

            if self.model_type == 'i2vgen' and self.guide_scale > 0.0:
                negative_task_embeds = [x[1] for x in task_embeds]
                task_embeds = [x[0] for x in task_embeds]
                # with self.accelerator.autocast():
                all_xs_list = list(map(lambda n, l, c, e: self.ema.ema_model.module.sample(batch_size=n, negative_task_emb=l, image=c, action_data=e), batches, negative_task_embeds, x_conds, task_embeds))  
            else:
                # with self.accelerator.autocast():
                all_xs_list = list(map(lambda n, c, e: self.ema.ema_model.module.sample(batch_size=n, image=c, action_data=e), batches, x_conds, task_embeds))
        

        print_gpu_utilization()
        
        gt_xs = torch.cat(xs, dim = 0) # [batch_size, 3*n, 120, 160]
        # make it [batchsize*n, 3, 120, 160]
        n_rows = gt_xs.shape[2]
        if self.model_type == 'avdc':
            n_rows = gt_xs.shape[1] // 3
            gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=n_rows)

        ### save images
        x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
        # x_conds = rearrange(x_conds, 'b (n c) h w -> b n c h w', n=1)
        all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
        if self.model_type == 'avdc':
            all_xs = rearrange(all_xs, 'b (n c) h w -> b n c h w', n=n_rows)
            gt_xs = gt_xs.detach().cpu()
            gt_first = gt_xs[:, :1]
            gt_last = gt_xs[:, -1:]
        else:
            gt_xs = gt_xs.detach().cpu()
            gt_first = gt_xs[:, :,  :1]
            gt_last = gt_xs[:, :, -1:]

        if self.step == self.save_and_sample_every:
            os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
            if self.model_type=='avdc': 
                gt_img = torch.cat([gt_first, gt_last, gt_xs], dim=1)
                gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', n=n_rows+2)
            else: 
                gt_img = torch.cat([gt_first, gt_last, gt_xs], dim=2)
                gt_img = rearrange(gt_img, 'b c n h w -> (b n) c h w', n=n_rows+2)
            utils.save_image(gt_img, str(self.results_folder / f'imgs/gt_img.png'), nrow=n_rows+2)

        os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
        if self.model_type == 'avdc': 
            pred_img = torch.cat([gt_first, gt_last,  all_xs], dim=1)
            pred_img = rearrange(pred_img, 'b n c h w -> (b n) c h w', n=n_rows+2)
        else: 
            pred_img = torch.cat([gt_first, gt_last,  all_xs], dim=2)
            pred_img = rearrange(pred_img, 'b c n h w -> (b n) c h w', n=n_rows+2)
            
        utils.save_image(pred_img, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=n_rows+2)

        self.save(milestone)

    def train(self):
        # accelerator = self.accelerator
        # device = accelerator.device
        device = self.gpu

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not self.main_process) as pbar:

            # if self.accelerator.is_main_process:
            if self.main_process:
                # self.eval()
                pass

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    batch = dict_to_device(batch, device)

                    x = batch['video']
                    goal_embed = self.encode_action(batch)
                    negative_embed = None
                    if self.model_type == 'i2vgen' and self.guide_scale > 0.0:
                        goal_embed, negative_embed = goal_embed
            
                    # with self.accelerator.autocast():
                    loss, model_pred = self.model(x, goal_embed, negative_task_emb=negative_embed)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()
                    
                    loss.backward()

                        # self.accelerator.backward(loss)

                # self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                # scale = self.accelerator.scaler.get_scale()
                
                pbar.set_description(f'loss: {total_loss:.4E}, loss scale: {scale:.1E}')

                # self.accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                # self.accelerator.wait_for_everyone()

                self.step += 1
                # if accelerator.is_main_process:
                if self.main_process:
                    self.ema.update()
                    if self.step != 0 and self.step % self.save_and_sample_every == 0: self.eval()

                pbar.update(1)

        # accelerator.print('training complete')
