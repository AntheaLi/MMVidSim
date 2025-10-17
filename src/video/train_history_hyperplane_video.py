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
import torch.nn.functional as F
from PIL import Image
from subprocess import call
import importlib

from subprocess import call

from data import ActionSenseVideo
from models.model_hyperplane_history_trainer import Trainer
from models.model_video_history_hyperplane import VideoDiffusion, Network
# from exp.action.models.encoders import TextEncoder
from models.encoders import TextEncoder, ImageEncoder
from models.model_modules import ResNetImageEncoder
import wandb
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    wandb.login(key='1ad0d8ed1b4e7f2ef84ab842961c1dcbcdecda8a')
    
    writer = wandb.init(project='mm-act', group=f'{args.video_model}-hyperplane-{args.hyperplane}', name=f'{args.video_model}-hyperplane-{args.hyperplane}')

    signals = ['myo-emg-left', 'myo-emg-right', 'tactile-glove-left', 'tactile-glove-right', 'right-hand-pose', 'left-hand-pose', 'joint-position']

    train_dataset = ActionSenseVideo(path=args.data_path, split='train', parse_signal_keys=signals, horizon=4, resample_len=args.data_resample_len, video_res=args.image_size)
    val_dataset = ActionSenseVideo(path=args.data_path, split='val', parse_signal_keys=signals, horizon=4, resample_len=args.data_resample_len, video_res=args.image_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # experiment logging and directory:
    start_time = datetime.now()
    exp_dir = f"./{args.log_dir}/{args.exp_suffix}-{str(start_time).replace(' ', '-')}/"
    best_ckpt = f'./{exp_dir}/ckpts/best_ckpt.pth'
    device = 'cuda:0'
    os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, 'ckpts'))
    os.makedirs(os.path.join(exp_dir, 'visu'))
    # os.system(f'cp data.py models/{args.model}.py {__file__} {exp_dir}')

    # create models
    args.signal_model_config = parse_signal_model_config(signals, args)
    model = Network(args, cond_drop_chance=args.cond_drop_chance, signals=signals)
    if args.pretrain != '': model._load_pretrained_signal_encoder(args.pretrain)
    text_model = TextEncoder(model_type=args.text_model)
    text_model = text_model.to(device)
    text_model.requires_grad_(False)
    text_model.eval()
    
    image_model = ImageEncoder(model_type=args.image_model)
    image_model = image_model.to(device)
    image_model.requires_grad_(False)
    image_model.eval()
    
    
    torch.save(args, f'{exp_dir}/ckpts/args.pt')

    trainer = Trainer(diffusion_model=model,
                      model_type=args.video_model,
                      condition_encoder=text_model, 
                      context_encoder=image_model, 
                      train_set=train_dataset,
                      valid_set=val_dataset,
                      wandb_writer=writer,
                      results_folder=exp_dir,
                      use_text_condition=False,
                      cond_drop_chance=args.cond_drop_chance,
                      train_batch_size=args.batch_size,
                      valid_batch_size=args.batch_size, 
                      guide_scale=args.guide_scale,
                      **args.train_config)


    # create logs
    header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR     Loss'

    if args.ckpt is not None:
        trainer.load(args.ckpt)
        print('loaded', args.ckpt)
    
    trainer.train()

    if args.inference:
        from PIL import Image
        from torchvision import transforms
        import imageio
        import torch
        from os.path import splitext
        text = args.text
        image = Image.open(args.inference_path)
        batch_size = 1
        transform = transforms.Compose([
            transforms.Resize(64, 64),
            transforms.ToTensor(),
        ])
        image = transform(image)
        output = trainer.sample(image.unsqueeze(0), [text], batch_size).cpu()
        output = output[0].reshape(-1, 3, 64, 64)
        output = torch.cat([image.unsqueeze(0), output], dim=0)
        root, ext = splitext(args.inference_path)
        output_gif = root + '_out.gif'
        output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
        imageio.mimsave(output_gif, output, duration=200, loop=1000)
        print(f'Generated {output_gif}')








