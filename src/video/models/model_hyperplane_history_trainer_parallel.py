

import os
import sys
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import wandb

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler

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

from exp.utils import dict_to_device, dict_to_float
from exp.utils import *


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


def collate_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        condition_encoder, 
        context_encoder,
        train_set,
        valid_set,
        wandb_writer=None,
        sampler=None,
        channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        model_type = 'avdc',
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 5000,
        num_epochs=10000,
        eval_freq=20,
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

        assert model_type in ['avdc', 'i2vgen', 'i2vgen-avdc', 'our', 'ours', 'ours-lcd', 'ours-lcd-ll', 'ours-i2v'], f'{model_type} not implemented'
        self.model_type = model_type

        self.use_context_action = True

        self.cond_drop_chance = cond_drop_chance

        self.wandb_writer = wandb_writer
        
        self.use_text_condition = use_text_condition

        self.guide_scale = guide_scale

        n_worker = 32

        if condition_encoder is None:
            assert not self.use_text_condition, f' not conditioning model passed in'
        # accelerator

        self.accelerator = Accelerator()

        # self.accelerator.native_amp = amp

        # model


        self.channels = channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            # self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples

        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.num_epochs = num_epochs
        self.eval_freq = eval_freq

        # dataset and dataloader
        valid_ind = [i for i in range(len(valid_set))][:num_samples]

        train_set = train_set
        valid_set = Subset(valid_set, valid_ind)

        self.ds = train_set
        self.valid_ds = valid_set
    
    
        # train_sampler = DistributedSampler(self.ds, shuffle=True)
        dl = DataLoader(self.ds, batch_size = train_batch_size, pin_memory = True, shuffle=True, num_workers = n_worker, collate_fn = collate_none)#, sampler=train_sampler)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, pin_memory = True, shuffle=False, num_workers = 32, collate_fn = collate_none, sampler=train_sampler)
        
        valid_dl = DataLoader(self.valid_ds, batch_size = valid_batch_size, shuffle = False, pin_memory = True, num_workers = n_worker, collate_fn = collate_none)
        self.num_batches =  (len(self.ds) // train_batch_size)
        self.eval_num_steps = int(self.eval_freq * self.num_batches)
        self.save_and_sample_every = self.eval_num_steps
        print('data len', len(self.ds),  'num_batches', self.num_batches, 'saving freq', self.eval_num_steps)


        # self.model = diffusion_model
        diffusion_model = diffusion_model.float()
        condition_encoder = condition_encoder.float()
        context_encoder = context_encoder.float()
        
        # optimizer
        optimizer = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        self.model, self.condition_encoder, self.context_encoder, self.opt, self.dl, self.valid_dl = self.accelerator.prepare(diffusion_model, condition_encoder, context_encoder, optimizer, dl, valid_dl)
        self.dl = cycle(self.dl)
        
        

        # if self.accelerator.is_main_process:
        self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
        # self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        # self.model, self.opt, self.condition_encoder = \
        #     self.accelerator.prepare(self.model, self.opt, self.condition_encoder)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(milestone, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            data['ema']['initted'] = data['ema']['initted'].reshape(1)
            data['ema']['step'] = data['ema']['step'].reshape(1)
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def load_context(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(milestone, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=False)

        self.step = data['step']
        if self.accelerator.is_main_process:
            data['ema']['initted'] = data['ema']['initted'].reshape(1)
            data['ema']['step'] = data['ema']['step'].reshape(1)
            self.ema.load_state_dict(data["ema"], strict=False)

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        self.opt.load_state_dict(data['opt'])


    def sample(self, batch, batch_condition, batch_size=1, image_feat=None, guide_scale=0):
        device = self.device
        task_embeds = self.encode_action(batch_condition)
        if image_feat is not None: self.ema.ema_model.sample(image=batch, action_data=task_embeds, batch_size=batch_size, image_feat=task_embeds['context'])
        return self.ema.ema_model.sample(image=batch, action_data=task_embeds, batch_size=batch_size)


    def sample_latent(self, batch, task_embeds, batch_size=1, image_feat=None, guide_scale=0):
        if image_feat is not None: self.ema.ema_model.sample_latent(image=batch, signal_feature=task_embeds, batch_size=batch_size, image_feat=task_embeds['context'])
        return self.ema.ema_model.sample_latent(image=batch, signal_feature=task_embeds, batch_size=batch_size)


    def encode_action(self, batch):
        if self.use_text_condition: 
            goal = batch['label_text']
            goal_embed = self.condition_encoder(goal)
            
            if self.model_type == 'ours-lcd':
                return goal_embed
            
            action = goal_embed * (torch.rand(goal_embed.shape[0], 1, 1, device = goal_embed.device) > self.cond_drop_chance)#.float()
            return action
        
        else:
            action = {x: batch['signal'][x][:, 4:4+8] for x in batch['signal'].keys()}
            if self.use_context_action: 
                context_action_z = {x: batch['signal'][x][:, :4] for x in batch['signal'].keys()}
                action['action_context'] = context_action_z

            if self.context_encoder is not None:
                context_img_z = []
                for i in range(4):
                    context_img_z.append(self.context_encoder(batch['video'][:, i]))
                # action['context'] = rearrange(torch.stack(context_img_z), 't b d -> b (t d)') # concatenate in channel dimmension
                action['history_context'] = torch.stack(context_img_z)
                action['context'] = torch.stack(context_img_z).mean(0) # avg pool over 
                action['video_latent'] = self.context_encoder(rearrange(batch['video'], 'b t c h w -> (b t) c h w'))
                
            if self.condition_encoder is not None:
                goal = batch['label_text']
                action['text'] = self.condition_encoder(goal)
            
            return action


    def eval(self):

        self.ema.ema_model.eval()

        with torch.no_grad():
            milestone = self.step // self.eval_num_steps
            # batches = num_to_groups(self.num_samples, self.valid_batch_size)
            ### get val_imgs from self.valid_dl
            x_conds = []
            xs = []
            task_embeds = []
            image_features = []
            
            batches = []
            
            for i, batch in enumerate(self.valid_dl):
                # batch = dict_to_device(batch, self.device)
                # batch = dict_to_device(batch, self.device)
                batch = dict_to_float(batch)
                
                val_goal_emb = self.encode_action(batch)
                task_embeds.append(val_goal_emb)

                if self.model_type == 'avdc':
                    x = batch['video'].permute(0, 1, 4, 2, 3)
                    b, t, c, h, w = x.shape
                    x_cond = x[:, 0:4]
                    x_conds.append(x_cond.reshape(b, 1 * c, h, w))
                    xs.append(x[:, 4:].reshape(b, (t - 1) * c, h, w))
                    

                elif 'i2vgen' in self.model_type or 'our' in self.model_type:
                    x = batch['video'].permute(0, 4, 1, 2, 3)
                    b, c, t, h, w = x.shape
                    print(b, c, t, h, w)
                    x_conds.append(x[:, :, 0:4])
                    xs.append(x[:, :, 4:])
                    if not self.use_text_condition: image_features.append(val_goal_emb['context']) 

                else:
                    raise NotImplementedError

                batches.append(b)


            if self.model_type == 'i2vgen' and self.guide_scale > 0.0:
                negative_task_embeds = [x[1] for x in task_embeds]
                task_embeds = [x[0] for x in task_embeds]
                with self.accelerator.autocast():
                    all_xs_list = list(map(lambda n, l, c, e: self.ema.ema_model.module.sample(batch_size=n, negative_task_emb=l, image=c, action_data=e), batches, negative_task_embeds, x_conds, task_embeds))  
            else:
                with self.accelerator.autocast():
                    all_xs_list = list(map(lambda n, c, e: self.ema.ema_model.module.sample(batch_size=n, image=c, action_data=e), batches, x_conds, task_embeds))
        

        print_gpu_utilization()
        
        gt_xs = torch.cat(xs, dim = 0) # [batch_size, 3*n, 120, 160]
        # make it [batchsize*n, 3, 120, 160]
        n_rows = gt_xs.shape[2]
        if self.model_type == 'avdc':
            n_rows = gt_xs.shape[1] // 3
            gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=12)

        ### save images
        x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
        # x_conds = rearrange(x_conds, 'b (n c) h w -> b n c h w', n=1)
        all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
        if self.model_type == 'avdc':
            all_xs = rearrange(all_xs, 'b (n c) h w -> b n c h w', n=12)
            gt_xs = gt_xs.detach().cpu()
            gt_first = gt_xs[:, :4]
            gt_img = torch.cat([gt_first, gt_xs], dim=1)
            gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', n=12)
        else:
            gt_xs = gt_xs.detach().cpu()
            gt_first = gt_xs[:, :,  :4]
            gt_img = torch.cat([gt_first, gt_xs], dim=2)
            gt_img = rearrange(gt_img, 'b c n h w -> (b n) c h w', n=12)

        if self.step == self.eval_num_steps:
            os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
            utils.save_image(gt_img, str(self.results_folder / f'imgs/gt_img.png'), nrow=12)
            
        os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
        if self.model_type == 'avdc': 
            pred_img = torch.cat([gt_first,  all_xs], dim=1)
            pred_img = rearrange(pred_img, 'b n c h w -> (b n) c h w', n=12)
        else: 
            pred_img = torch.cat([gt_first,  all_xs], dim=2)
            pred_img = rearrange(pred_img, 'b c n h w -> (b n) c h w', n=12)
            
        utils.save_image(pred_img, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=12)

        # batch_size = gt_first.shape[0]
        # if batch_size == 1:
        #     rearrange_order = 'n c h w -> b c h (n w)'
        # else:
        #     rearrange_order = '(b n) c h w -> b c h (n w)'
        
        
        self.wandb_writer.log({"gt-img": [wandb.Image(gt_img[i].permute(1, 2, 0).detach().cpu().numpy(), caption="GroundTruth") for i in range(12)]})
        self.wandb_writer.log({"pred-img": [wandb.Image(gt_img[i].permute(1, 2, 0).detach().cpu().numpy(), caption="GroundTruth") for i in range(12)]})
        
        pred_img = rearrange(pred_img, '(b n) c h w -> b c h (n w)', n=12)
        self.wandb_writer.log({"sample-v": [wandb.Image(pred_img[i].permute(1, 2, 0).detach().cpu().numpy(), caption="Generated") for i in range(pred_img.shape[0])]})
        gt_img = torch.cat([gt_first, gt_xs], dim=2)
        self.wandb_writer.log({"gt-v": [wandb.Image(gt_img[i].permute(1, 2, 0).detach().cpu().numpy(), caption="GT") for i in range(gt_img.shape[0])]})
        
        

        
        if self.step > 0:
            print('saving check point from step', self.step)
            self.save(milestone)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        # , disable = not accelerator.is_main_process

        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:

            # if accelerator.is_main_process:
            #     self.eval()

            while self.step < self.train_num_steps:

                total_loss = 0.

                with torch.autograd.set_detect_anomaly(True):
                    for _ in range(self.gradient_accumulate_every):
                        batch = next(self.dl)
                        # batch = dict_to_device(batch, self.device)
                        batch = dict_to_float(batch)
                        
                        x = batch['video']
                        goal_embed = self.encode_action(batch)
                        negative_embed = None
                        if self.model_type == 'i2vgen' and self.guide_scale > 0.0:
                            goal_embed, negative_embed = goal_embed
                
                        with self.accelerator.autocast():
                            if self.condition_encoder is not None: text_feature = self.condition_encoder(batch['label_text'])
                            else: text_feature = None
                            x_dist = batch['video_dist']
                            loss, model_pred = self.model(x, goal_embed, x_dist, text_feature, negative_task_emb=negative_embed)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                            self.accelerator.backward(loss)

                        for name, param in self.model.module.named_parameters():
                            if param.grad is None:
                                print(name)

                    self.accelerator.clip_grad_norm_(self.model.module.parameters(), 1.0)

                    # scale = self.accelerator.scaler.get_scale()
                    
                    log_str = f'loss: {total_loss:.4E}'
                    if self.accelerator.scaler: log_str += f', loss scale: {scale:.1E}' 
                    
                    pbar.set_description(log_str)
                    self.wandb_writer.log({"loss": total_loss})

                    self.accelerator.wait_for_everyone()

                    self.opt.step()
                    self.opt.zero_grad()

                    self.accelerator.wait_for_everyone()

                    self.step += 1
                    if accelerator.is_main_process:
                        self.ema.update()
                        if self.step != 0 and self.step % self.eval_num_steps == 0: self.eval()

                    pbar.update(1)

                    # torch.cuda.empty_cache()

        accelerator.print('training complete')

    def train_epoch(self):
        accelerator = self.accelerator
        device = accelerator.device


        self.eval()

        # while self.step < self.train_num_steps:

        for step in self.num_epochs: 

            total_loss = 0.

            pbar = tqdm(self.dl, leave=False)

            pbar.set_description(f'[{step}/{self.num_epochs}]')


            # train for every batch
            for train_batch_ind, batch in enumerate(pbar):
                # batch = dict_to_device(batch, self.device)
                batch = dict_to_float(batch)

                x = batch['video']
                goal_embed = self.encode_action(batch)
                negative_embed = None
                if self.model_type == 'i2vgen' and self.guide_scale > 0.0:
                    goal_embed, negative_embed = goal_embed
        
                with self.accelerator.autocast():
                    text_feature = self.condition_encoder(batch['label_text'])
                    x_dist = batch['video_dist']
                    loss, model_pred = self.model(x, goal_embed, x_dist, text_feature, negative_task_emb=negative_embed)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    self.accelerator.backward(loss)

                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        print(name)

                self.accelerator.clip_grad_norm_(self.model.module.parameters(), 1.0)

                scale = self.accelerator.scaler.get_scale()
                
                pbar.set_description(f'loss: {total_loss:.4E}, loss scale: {scale:.1E}')

                self.accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                self.accelerator.wait_for_everyone()

            self.step = step
            if accelerator.is_main_process:
                self.ema.update()
                if self.step != 0 and self.step % self.eval_freq == 0: self.eval()

        accelerator.print('training complete')
