from argparse import ArgumentParser
import torch
from torch import nn
import json
import yaml
from easydict import EasyDict
from action_config import signal_model_cfg

import sys; sys.path.append('../')


cfg = EasyDict()

img_size = (64, 64)
# img_size = (224, 224)

signal_model_cfg['p_zero'] = 0.0

double_z = True

num_frames = 8

text_model_cfg = {
    'clip': {'version': "openai/clip-vit-large-patch14",
             'layer': 'last', # hidden / last / pooled
             'layer_idx': 11, # from 0 ~ 12
             },
    'open_clip': {'arch': 'ViT-H-14',
                  'version':"laion2b_s32b_b79k", 
                  'layer_idx': 0, # 0: last or 1: penultimate
                  }, #"ViT-H-14"
    't5': {'version':"google/t5-base"},
    'byt5': {'version':"google/byt5-base"},
    'llama': {'version':'huggyllama/llama-7b'}, #meta-llama/Llama-2-7b-hf'
    'layer': 'last', #penultimate
    'p_zero': 0.1,
}

image_model_cfg = {
    'clip': {'version': "openai/clip-vit-large-patch14",
             'layer': 'last', # hidden / last / pooled
             'layer_idx': 11, # from 0 ~ 12
             'dimension': 784
             },
    'open_clip': {'arch': 'ViT-H-14',
                  'version':"laion2b_s32b_b79k", 
                  'layer_idx': 0, # 0: last or 1: penultimate
                  'dimension':1024,
                  }, #"ViT-H-14"
    'resnet': {'dimension':2048},
    'resnet-train': {'dimension':2048},
    'layer': 'last', #penultimate
    'p_zero': 0.1,
}

avdc_setup = {
    'channels': 3 * (10 - 1),
    'image_size': img_size,
    'timesteps':1000,
    'sampling_timesteps': 1000,
    'loss_types': ['l2'],
    'objective':'pred_v',
    'beta_schedule':'cosine',
    'min_snr_loss_weight': True}

i2v_avdc_setup = {
    'channels': 3,
    'num_frames': num_frames,
    'image_size': img_size,
    'timesteps':1000,
    'sampling_timesteps': 1000,
    'loss_types': ['l2'],
    'objective':'pred_v',
    'beta_schedule':'cosine',
    'min_snr_loss_weight': True}

i2v_ae_setup = {
    'ddconfig': {
        'double_z': True, 
        'z_channels': 4,
        'resolution': 64, # check this 
        'in_channels': 3,
        'out_ch': 3, 
        'ch': 128, 
        'ch_mult': [1, 2, 4, 4],
        'num_res_blocks': 2, 
        'attn_resolutions': [], 
        'dropout': 0.0,
        'video_kernel_size': [3, 1, 1]
    },
    'embed_dim': 4,
}

i2v_unet = {
    'in_dim': 3,
    'dim': 64, 
    'y_dim': 1024,
    'context_dim': 1024,
    'frames': num_frames, 
    'out_dim': 3,
    'concate_dim': 3,
    'dim_mult': [1, 2, 4, 4],
    'num_heads': 8,
    'head_dim': 64,
    'num_res_blocks': 2,
    'dropout': 0.1,
    'temporal_attention': True,
    'temporal_attn_times': 1,
    'use_checkpoint': True,
    'use_fps_condition': False,
    'use_sim_mask': False
}

i2v_diffusion = {
    'image_size': img_size,
    'channels': 3, 
    'num_frames': num_frames,
    'beta_schedule': 'cosine', # cosine
    'schedule_param': {
        'num_timesteps': 1000,
        'cosine_s': 0.008,
        'zero_terminal_snr': True,
    },
    'mean_type': 'v',
    'loss_type': 'mse',
    'var_type': 'fixed_small',
    'rescale_timesteps': False,
    'noise_strength': 0.1,
    'ddim_sampling': True,
    'fps': [8,  8,  16, 16, 16, 8,  16, 16]
}

i2v_setup = {'unet':i2v_unet,
       'diffusion':i2v_diffusion,}

our_unet = {
    'in_dim': 3,
    'dim': 64, 
    'y_dim': 1024,
    'context_dim': 1024,
    'frames': num_frames, 
    'out_dim': 3,
    'concate_dim': 3,
    'dim_mult': [1, 2, 4, 4],
    'num_heads': 8,
    'head_dim': 64,
    'num_res_blocks': 2,
    'dropout': 0.1,
    'temporal_attention': True,
    'temporal_attn_times': 1,
    'use_checkpoint': True,
    'use_fps_condition': False,
    'use_sim_mask': False
}

our_unet_lcd = {
    'in_dim': 3,
    'dim': 64, 
    'y_dim': 1024,
    'context_dim': 1024,
    'frames': num_frames, 
    'out_dim': 3,
    'concate_dim': 3,
    'dim_mult': [1, 2, 4, 4],
    'num_heads': 8,
    'head_dim': 64,
    'num_res_blocks': 2,
    'dropout': 0.1, 
    'temporal_attention': True,
    'temporal_attn_times': 1,
    'use_checkpoint': True,
    'use_fps_condition': False,
    'use_sim_mask': False,
    'image_model_choice': 'resnet-train', 
}

our_setup = {'unet':our_unet,
             'unet-lcd': our_unet_lcd,
            'diffusion':i2v_avdc_setup,}

signal_model_cfg['pool'] = 'max'

sd_unet = {
    'adm_in_channels': 768,
    'num_classes': 'sequential',
    'use_checkpoint': True,
    'in_channels': 8,
    'out_channels': 4,
    'model_channels': 320,
    'attention_resolutions': [4, 2, 1],
    'num_res_blocks': 2,
    'channel_mult': [1, 2, 4, 4],
    'num_head_channels': 64,
    'use_linear_in_transformer': True,
    'transformer_depth': 1,
    'context_dim': 1024,
    'spatial_transformer_attn_type': 'softmax-xformers',
    'extra_ff_mix_layer': True,
    'use_spatial_context': True,
    'merge_strategy': 'learned_with_images',
    'video_kernel_size': [3, 1, 1],
}

sd_diffusion = {
    'channels': 3,
    'num_frames': num_frames,
    'image_size': img_size,
    'timesteps':1000,
    'sampling_timesteps': 1000,
    'loss_types': ['l2'],
    'objective':'pred_v',
    'beta_schedule':'cosine',
    'min_snr_loss_weight': True}

sd_avdc_setup = {
    'unet': sd_unet,
    'diffusion': sd_diffusion,
}

sd_ae_setup = {
    'encoder':{
        'attn_type' : 'vanilla',
        'double_z': double_z,
        'z_channels': 4,
        'resolution': 64,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 4, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0,
    },
    'decoder':{
        'attn_type': 'vanilla',
        'double_z': double_z,
        'z_channels': 4,
        'resolution': 64,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 4, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0,
        'video_kernel_size': [3, 1, 1],
    }
}

video_model_cfg = {
    'avdc': avdc_setup,
    'i2vgen': i2v_setup, 
    'i2vgen-avdc': i2v_avdc_setup, 
    'our': our_setup, 
    'i2vgen-xl': {'verision': "ali-vilab/i2vgen-xl"},
    'aa': {'unet': 'placeholder'},
    'dtype': torch.float16,
    'image_size': img_size,
    'guide_scale': 9.0, 
    'num_frames': num_frames
}

trainer_config = {
    'gradient_accumulate_every':1,
    'augment_horizontal_flip' :True,
    'train_lr' :1e-4,
    'train_num_steps': 1000000,
    'ema_update_every' :10,
    'ema_decay' :0.999,
    'save_and_sample_every': 4000,
    'num_samples': 6,
    'amp':True,
    'fp16':True,
    'split_batches': True,
    'convert_image_to' :None,
    'calculate_fid' :True,
    }


def parse_args():
    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--model', type=str, help='model def file')
    parser.add_argument('--exp-suffix', type=str, default='test', help='exp suffix')
    parser.add_argument('--data-path', type=str, help='data path', default='./Dataset/')
    parser.add_argument('--signal', type=str, default='tactile-glove-left tactile-glove-right myo-left myo-right joint-rotation joint-position right-hand-pose left-hand-pose')


    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (for reproducibility) [specify -1 means to generate a random one]') # 
    parser.add_argument('--log-dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--ckpt', type=str, help='load checkpoint', default=None)


    # network settings
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--ae-model', type=str, default='sd', choices=['i2vgen', 'i2vgen-xl', 'avdc', 'i2vgen-avdc', 'sd',])
    parser.add_argument('--lcd-ctx-model', type=str, default='resnet-train', choices=['non-resnet-train', 'resnet-train'])
    parser.add_argument('--video-model', type=str, default='avdc', choices=['aa', 'i2vgen', 'i2vgen-xl', 'avdc', 'i2vgen-avdc', 'sd', 'sd-avdc', 'ours', 'our', 'ours-lcd', 'our-lcd', 'ours-i2v', 'our-i2v'])
    parser.add_argument('--text-model', type=str, default='open_clip', choices=['t5', 'open_clip', 'clip', 'llama'])
    parser.add_argument('--image-model', type=str, default='resnet-train', choices=['open_clip', 'clip', 'resnet', 'resnet-train'])
    parser.add_argument('--tactile-model', type=str, default='resnet', choices=['mae', 'vivit', 'neuralfield', 'conv', 'mvit', 'resnet'])
    parser.add_argument('--joint-position-model', type=str, default='mlp-frame', choices=['transformer', 'mlp', 'neuralfield', 'mlp-frame'])
    parser.add_argument('--joint-rotation-model', type=str, default='mlp-frame', choices=['transformer', 'mlp', 'neuralfield', 'mlp-frame'])
    parser.add_argument('--hand-pose-model', type=str, default='mlp-frame', choices=['transformer', 'mlp', 'neuralfield', 'mlp-frame'])
    parser.add_argument('--myo-model', type=str, default='mlp-frame', choices=['transformer', 'mlp', 'neuralfield', 'mlp-frame'])
    parser.add_argument('--hyperplane', type=str, default='individual', choices=['global', 'individual', 'global-metric', 'global-metric-linear', 'global-metric-mlp', 'local-metric-mlp'])
    parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
    parser.add_argument('--merge-local', type=str, default='sum', choices=['max', 'sum', 'cat', 'cross', 'maxpool', 'softmax'])
    parser.add_argument('--skip', action='store_true', default=False)
    


    # training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--signal-lr', type=float, default=1e-3)
    parser.add_argument('--text-lr', type=float, default=1e-7)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--lr-decay-by', type=float, default=0.9)
    parser.add_argument('--lr-decay-every', type=float, default=5000)
    parser.add_argument('--guide-scale', type=float, default=0.0)
    parser.add_argument('--inference', action='store_true', default=False)
    
    parser.add_argument('--pretrain', type=str, default='')
    
    # loss weights
    parser.add_argument('--loss-weight-tactile', type=float, default=1.0, help='loss weight')

    # logging
    parser.add_argument('--no-tb', action='store_true', default=False)
    parser.add_argument('--no-console-log', action='store_true', default=False)
    parser.add_argument('--eval-freq', type=int, default=4)


    # visu
    parser.add_argument('--world_size', default=1, type=int,help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,help='node rank for distributed training')

    parser.add_argument('--local-rank', type=int, default=0)

    # data
    parser.add_argument('--data-resample-len', type=int, default=9, help='data resample len')

    # jupyter 
    parser.add_argument('--f', type=str, default='')
    
    # parse args
    args = parser.parse_args()

    video_model_cfg['guide_scale'] = args.guide_scale
    video_model_cfg['our']['unet-lcd']['image_model_choice'] = args.lcd_ctx_model
    args.train_config = trainer_config
    args.train_config['num_epochs'] = args.epochs
    args.train_config['eval_freq'] = args.eval_freq
    args.text_model_config = text_model_cfg
    args.image_model_config = image_model_cfg
    args.video_model_config = video_model_cfg
    args.signal_model_config = signal_model_cfg
    args.signal_model_pool = 'max'
    args.cond_drop_chance = 0.0
    args.image_size = img_size[0]

    return args


def parse_signal_model_config(signals, args):
    signal_model_config = {}
    
    for sig in signals:
        if 'tactile' in sig: 
            signal_model_config[sig] = args.tactile_model
        elif 'myo' in sig: 
            signal_model_config[sig] = args.myo_model
        elif 'hand-pose' in sig: 
            signal_model_config[sig] = args.hand_pose_model
        elif 'joint-position' in sig: 
            signal_model_config[sig] = args.joint_position_model
        elif 'joint-rotation' in sig: 
            signal_model_config[sig] = args.joint_rotation_model
        else:
            raise NotImplementedError
        
    return signal_model_config




