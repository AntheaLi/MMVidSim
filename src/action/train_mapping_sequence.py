import os
import time
import sys
import shutil
import random
from copy import deepcopy
from tqdm import tqdm
from time import strftime
from itertools import chain
from config import parse_args, parse_signal_model_config

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from subprocess import call
import importlib
from tensorboardX import SummaryWriter

from exp.utils import *
from data import ActionSenseCaption



def train_one_epoch(args, model, text_encoder,  train_dataloader, optimizers, train_writer, epoch):
    model.train()
    text_encoder.eval()
    train_num_batch = len(train_dataloader)
    pbar = tqdm(train_dataloader, leave=False)

    sum_total_loss = 0
    pbar.set_description(f'[{epoch}/{args.epochs}]')


    # train for every batch
    for train_batch_ind, batch in enumerate(pbar):
        for opt in optimizers: opt.zero_grad()

        batch = dict_to_device(batch, device=args.device)
        encoded_text = text_encoder(batch['label_text'])
        loss_dict, loss = network(batch, encoded_text)

        # optimize one step
        network_lr_scheduler.step()

        loss.backward()
        sum_total_loss += loss.item()

        for opt in optimizers: opt.step()
    
        pbar.set_postfix({'Loss': loss.item()})

        if train_batch_ind == 0: sum_loss_dict = {k:loss_dict[k].item() for k in loss_dict.keys()}
        else: sum_loss_dict = update_dict(loss_dict, sum_loss_dict)
        # print(f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} [train]  {epoch:>5.0f}/{args.epochs:<5.0f}  {train_batch_ind:>5.0f}/{train_num_batch:<5.0f}      {100. * (1+train_batch_ind+train_num_batch*epoch) / (train_num_batch*args.epochs):>9.1f}%     {optimizers[0].param_groups[0]['lr']:>5.2E}        {loss.item():>10.5f}   ''')


    with torch.no_grad():
        train_writer.add_scalar('sum_total_train_loss', sum_total_loss, epoch)
        train_writer.add_scalar('lr', optimizers[0].param_groups[0]['lr'], epoch)
        train_writer.add_scalars('sum_signal_loss', sum_loss_dict, epoch)




def eval_one_epoch(args, model, text_encoder, val_dataloader, val_writer, epoch):
    model.eval()
    text_encoder.eval()
    val_num_batch = len(val_dataloader)
    pbar = tqdm(val_dataloader, leave=False)

    val_sum_total_loss_val = 0


    with torch.no_grad():

        for val_batch_ind, batch in enumerate(pbar):

            batch = dict_to_device(batch, device=args.device)

            encoded_text = text_encoder(batch['label_text'])
            loss_dict, loss = network(batch, encoded_text)

            val_sum_total_loss_val += loss.item()

            if val_batch_ind == 0: val_sum_loss_dict = {k:loss_dict[k].item() for k in loss_dict.keys()}
            else: val_sum_loss_dict = update_dict(loss_dict, val_sum_loss_dict)

        val_writer.add_scalar('val_sum_total_train_loss', val_sum_total_loss_val, epoch)
        val_writer.add_scalars('val_sum_signal_loss', val_sum_loss_dict, epoch)

    pbar.set_postfix({'Loss': val_sum_total_loss_val})

    # think about some visualization function to add here: #TODO
    print(f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} [val] {epoch:>5.0f}/{args.epochs:<5.0f}    {val_sum_total_loss_val:>10.5f}''')
    
    return loss



if __name__ == '__main__':
    args = parse_args()

    signals = ['myo-emg-left', 'myo-emg-right', 'tactile-glove-left', 'tactile-glove-right', 'right-hand-pose', 'left-hand-pose', 'joint-position']

    train_dataset = ActionSenseCaption(path=args.data_path, split='train', parse_signal_keys=signals, resample_len=args.data_resample_len)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_dataset = ActionSenseCaption(path=args.data_path, split='val', parse_signal_keys=signals, resample_len=args.data_resample_len)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=min(args.batch_size//2, len(val_dataset)-1), shuffle=False, num_workers=0, drop_last=True)


    # experiment logging and directory:
    start_time = time.time()
    exp_dir = f'./{args.log_dir}/{args.exp_suffix}-{start_time}/'
    best_ckpt = f'./{exp_dir}/ckpts/best_ckpt.pth'
    os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, 'ckpts'))
    os.makedirs(os.path.join(exp_dir, 'visu'))
    os.system(f'cp data.py models/{args.model}.py {__file__} {exp_dir}')

    # create models
    args.signal_model_config = parse_signal_model_config(signals, args)
    importlib.invalidate_caches()
    model_def = importlib.import_module('models.' + args.model)
    network = model_def.Network(args, signals=signals)
    text_model = model_def.TextEncoder(model_type=args.text_model)
    network.to(args.device)
    text_model.to(args.device)

    optimizer_signal = torch.optim.Adam(chain(network.signal_encoder.parameters(), network.signal_projection.parameters()), lr=args.signal_lr, weight_decay=args.weight_decay)
    optimizers = [optimizer_signal]
    optimizer_names = ['signal_optimizer']
    models = [network]
    model_names = [args.model]

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_signal, step_size=args.lr_decay_every, gamma=args.lr_decay_by)
    train_writer = SummaryWriter(os.path.join(exp_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(exp_dir, 'val'))

    # create logs
    header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR     Loss'
    

    # start training
    best_loss = eval_one_epoch(args, model=network, text_encoder=text_model, val_dataloader=val_dataloader, val_writer=val_writer, epoch=0)

    # train for every epoch
    for epoch in range(args.epochs):

        train_one_epoch(args, model=network, text_encoder=text_model, train_dataloader=train_dataloader,  optimizers=optimizers, train_writer=train_writer, epoch=epoch)

        if epoch % args.eval_freq == 0:
            test_loss = eval_one_epoch(args, model=network, text_encoder=text_model, val_dataloader=val_dataloader, val_writer=val_writer, epoch=epoch)

            if test_loss < best_loss:
                torch.save({'epoch': epoch + 1, 'model_state_dict': network.state_dict(),
                'signal_optim': optimizer_signal.state_dict(), 'args': vars(args)}, best_ckpt)
                 

            save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(exp_dir, 'ckpts'), \
            epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)


    # save the final models
    print('Saving final checkpoint ...... ')









