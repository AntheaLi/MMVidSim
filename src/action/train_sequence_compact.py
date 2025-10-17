import os
import time
import sys
import shutil
import random
from tqdm import tqdm
from time import strftime
from itertools import chain
from config import parse_args

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


def train_one_epoch(args, model, dataloader, optimizers, tb, epoch):
    model.train()
    train_num_batch = len(train_dataloader)
    pbar = tqdm(train_dataloader, leave=False)

    sum_total_loss = 0

    # train for every batch
    for train_batch_ind, batch in enumerate(pbar):
        for opt in optimizers: opt.zero_grad()

        batch = dict_to_device(batch, device=args.device)
        loss = network(batch)

        sum_total_loss += loss

        # optimize one step
        network_lr_scheduler.step()
        loss.backward()
        for opt in optimizers: opt.step()
    
        pbar.set_postfix({'Loss': loss.item()})
        # print(f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} [train]  {epoch:>5.0f}/{args.epochs:<5.0f}  {train_batch_ind:>5.0f}/{train_num_batch:<5.0f}      {100. * (1+train_batch_ind+train_num_batch*epoch) / (train_num_batch*args.epochs):>9.1f}%     {optimizers[0].param_groups[0]['lr']:>5.2E}        {loss.item():>10.5f}   ''')


    train_writer.add_scalar('sum_total_train_loss', sum_total_loss, epoch)
    train_writer.add_scalar('lr', optimizers[0].param_groups[0]['lr'], epoch)




def eval_one_epoch(args, model, dataloader, tb, epoch):
    model.eval()
    val_num_batch = len(val_dataloader)
    pbar = tqdm(val_dataloader, leave=False)

    sum_total_loss_val = 0

    for val_batch_ind, batch in enumerate(pbar):

        batch = dict_to_device(batch, device=args.device)
        loss = network(batch)

        sum_total_loss_val += loss.item()

        # using tensorboard to record the losses for each epoch
    with torch.no_grad():
        tb.add_scalar('sum_total_val_loss', sum_total_loss_val, epoch)

    pbar.set_postfix({'Loss': sum_total_loss_val})

    # think about some visualization function to add here: #TODO
    print(f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} [val] {epoch:>5.0f}/{args.epochs:<5.0f}    {sum_total_loss_val:>10.5f}''')
    
    return loss



if __name__ == '__main__':
    args = parse_args()

    signals = ['tactile', 'myo', 'joint-rotation', 'joint-position', 'hand-pose']

    train_dataset = ActionSenseCaption(parse_signal_keys=signals, resample_len=args.data_resample_len, compact=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=filter_none, shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_dataset = ActionSenseCaption(parse_signal_keys=signals, resample_len=args.data_resample_len, compact=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=min(args.batch_size//2, len(val_dataset)-1), collate_fn=filter_none, shuffle=False, num_workers=0, drop_last=True)


    # experiment logging and directory:
    start_time = time.time()
    exp_dir = f'./{args.log_dir}/{args.exp_suffix}-{start_time}/'
    best_ckpt = f'./{exp_dir}/ckpts/best_ckpt.pth'
    os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, 'ckpts'))
    os.makedirs(os.path.join(exp_dir, 'visu'))
    os.system(f'cp data.py models/{args.model}.py {__file__} {exp_dir}')

    # create models
    importlib.invalidate_caches()
    model_def = importlib.import_module('models.' + args.model)
    network = model_def.Network(args, signals=signals)
    network.to(args.device)

    optimizer_signal = torch.optim.Adam(chain(network.signal_encoder.parameters(), network.signal_projection.parameters()), lr=args.signal_lr, weight_decay=args.weight_decay)
    optimizer_text = torch.optim.Adam(chain(network.text_encoder.parameters(), network.text_projection.parameters()), lr=args.text_lr, weight_decay=args.weight_decay)
    optimizers = [optimizer_signal, optimizer_text]
    optimizer_names = ['signal_optimizer', 'text_optimizer']
    models = [network]
    model_names = [args.model]

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_signal, step_size=args.lr_decay_every, gamma=args.lr_decay_by)
    train_writer = SummaryWriter(os.path.join(exp_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(exp_dir, 'val'))

    # create logs
    header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR     Loss'

    # start training
    best_loss = eval_one_epoch(args, model=network, dataloader=val_dataloader, tb=val_writer, epoch=0)

    # train for every epoch
    for epoch in range(args.epochs):

        train_one_epoch(args, model=network, dataloader=train_dataloader,  optimizers=optimizers, tb=train_writer, epoch=epoch)

        if epoch % args.eval_freq == 0:
            test_loss = eval_one_epoch(args, model=network, dataloader=val_dataloader, tb=val_writer, epoch=epoch)

            if test_loss < best_loss:
                torch.save({'epoch': epoch + 1, 'model_state_dict': network.state_dict(),
                'signal_optim': optimizer_signal.state_dict(), 'text_optim': optimizer_text.state_dict(), 'args': vars(args)}, best_ckpt)
                 
            save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(args.exp_dir, 'ckpts'), \
            epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)


    # save the final models
    print('Saving final checkpoint ...... ')









