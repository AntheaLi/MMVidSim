from torch import nn
from argparse import ArgumentParser, Namespace

num_frames = 9
signal_latent_dim = 128
    

class BaseArgs(Namespace):
    hidden_size=256
    text_model='open_clip'
    tactile_model='resnet'
    joint_position_model='mlp-frame'
    joint_rotation_model='mlp-frame'
    hand_pose_model='mlp-frame'
    myo_model='mlp-frame'
    pool='avg'

    epochs=1000
    batch_size=32
    num_workers=8
    lr=1e-3
    head_lr=1e-3
    signal_lr=1e-3
    text_lr=1e-7
    weight_decay=1e-3
    lr_decay_by=0.9
    lr_decay_every=5000
    patience=1





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
    'layer': 'last'
}


joint_rotation_transformer_config = {
    "input_dim": 66,
    "kernel_size": 3,
    "stride": 3,
    "output_dim": 256,
    "num_layer": 2,
    "activation": 'gelu', 
    "device": 'cuda:0'}

joint_position_transformer_config = {
    "input_dim": 28 * 3,
    "kernel_size": 3, 
    "stride": 3,
    "output_dim": 256,
    "activation": 'gelu', 
    "num_layer": 2,
    "device": 'cuda:0'}

emg_transformer_config = {
    "input_dim": 8,
    "kernel_size": 3,
    "stride": 3,
    "output_dim": 256,
    "num_layer": 2,
    "activation": 'gelu', 
    "device": 'cuda:0'}

hand_pose_transformer_config = {
    "input_dim": 24 * 3,
    "kernel_size": 3,
    "stride": 3,
    "output_dim": 256,
    "num_layer": 2,
    "activation": 'gelu', 
    "device": 'cuda:0'
}

hand_pose_mlp_config = {
    "input_dim": 24 * 3,
    "hidden_dim": signal_latent_dim,
    "latent_dim": signal_latent_dim,
}


hand_pose_mlp_decoder_config = {
    "input_dim": signal_latent_dim,
    "hidden_dim": signal_latent_dim,
    "latent_dim": 24 * 3,
}

emg_mlp_config = {
    "input_dim": 8,
    "hidden_dim": signal_latent_dim,
    "latent_dim": signal_latent_dim,
}


emg_mlp_decoder_config = {
    "input_dim": signal_latent_dim,
    "hidden_dim": signal_latent_dim,
    "latent_dim": 8,
}

joint_rotation_mlp_config = {
    "input_dim": 22 * 3,
    "hidden_dim": signal_latent_dim,
    "latent_dim": signal_latent_dim,
}

joint_rotation_mlp_decoder_config = {
    "input_dim": signal_latent_dim,
    "hidden_dim": signal_latent_dim,
    "latent_dim": 22 * 3,
}

joint_position_mlp_config = {
    "input_dim": 28 * 3,
    "hidden_dim": signal_latent_dim,
    "latent_dim": signal_latent_dim,
}

joint_position_mlp_decoder_config = {
    "input_dim": signal_latent_dim,
    "hidden_dim": signal_latent_dim,
    "latent_dim": 28 * 3,
}


tactile_mae_config = {
    'image_size': 32,
    'num_frames': num_frames,
    'tubelet_size': 2,
    'patch_size': 4,
    'num_channels': 1,
    'hidden_size': 128,
    'num_hidden_layers': 3,
    'num_attention_heads': 2,
    'intermediate_size': 3072}


tactile_conv_config = {
    "input_channel": 1,
    "activation": nn.GELU}


tactile_conv_decoder_config = {
    'out_channels': 1,
    'in_features': 128,
    # 'spatial_size': (32, 32),
    # 'temporal_size': num_frames, 
}

tactile_mvit_config = {
    'spatial_size': (32, 32),
    'temporal_size': num_frames, 
    'depth': 16,
    'conv_patch_embed_kernel': (5, 4, 4),
    'conv_patch_embed_stride': (5, 4, 4),
    'conv_patch_embed_padding': (0, 0, 0),
    'input_channels': 1,
    'patch_embed_dim': 64, 
    'head': None,
}

tactile_vivit_config = {
    'image_size': 32,
    'num_frames': num_frames,
    'tubelet_size': [4, 16, 16],
    'num_channels': 1,
    'hidden_size': 64 * 5,
    'num_hidden_layers': 3,
    'num_attention_heads': 5,
    'intermediate_size': 3072}

joint_position_encoder_config = {
    'mlp': joint_position_mlp_config,
    'mlp-frame': joint_position_mlp_config,
    'transformer': joint_position_transformer_config,
    'neuralfield':{}
}

joint_rotation_encoder_config = {
    'mlp': joint_rotation_mlp_config,
    'mlp-frame': joint_rotation_mlp_config,
    'transformer': joint_rotation_transformer_config,
    'neuralfield':{}
}

handpose_encoder_config = {
    'mlp': hand_pose_mlp_config,
    'mlp-frame': hand_pose_mlp_config,
    'transformer': hand_pose_transformer_config,
    'neuralfield':{}
}

emg_encoder_config = {
    'mlp': emg_mlp_config,
    'mlp-frame': emg_mlp_config,
    'transformer': emg_transformer_config,
    'neuralfield': {},
}

tactile_encoder_config = {
    'mae': tactile_mae_config,
    'vivit':tactile_vivit_config,
    'conv': tactile_conv_config,
    'mvit':tactile_mvit_config,
    'resnet':{}, 
    'neuralfield': {},
}

joint_position_decoder_config = {
    'mlp': joint_position_mlp_decoder_config,
    'mlp-frame': joint_position_mlp_decoder_config,
    'neuralfield':{}
}

joint_rotation_decoder_config = {
    'mlp': joint_rotation_mlp_decoder_config,
    'mlp-frame': joint_rotation_mlp_decoder_config,
    'neuralfield':{}
}

handpose_decoder_config = {
    'mlp': hand_pose_mlp_decoder_config,
    'mlp-frame': hand_pose_mlp_decoder_config,
    'neuralfield':{}
}

emg_decoder_config = {
    'mlp': emg_mlp_decoder_config,
    'mlp-frame': emg_mlp_decoder_config,
    'neuralfield': {},
}

tactile_decoder_config = {
    'conv': tactile_conv_decoder_config,
}

signal_model_cfg = {
    'tactile': tactile_encoder_config,
    'myo': emg_encoder_config, 
    'joint-position': joint_position_encoder_config,
    'hand-pose': handpose_encoder_config,
    'joint-rotation': joint_rotation_encoder_config,
    'p_zero': 0.1,
}


signal_model_decoder_cfg = {
    'tactile': tactile_decoder_config, 
    'myo': emg_decoder_config,
    'joint-position': joint_position_decoder_config,
    'hand-pose': handpose_decoder_config,
    'joint-rotation': joint_rotation_decoder_config,
    'p_zero': 0.0,
}

loss_weight_config = {
    'tactile-glove-left': 1.0,
    'tactile-glove-right': 1.0,
    'myo-emg-left': 1.0, 
    'myo-emg-right': 1.0,
    'joint-position': 1.0,
    'left-hand-pose': 1.0,
    'right-hand-pose': 1.0,
    'joint-rotation': 1.0, 
}

tactile_projection_config = {
    'mvit': 8256,
    'conv': signal_latent_dim * num_frames,
    'vivit': signal_latent_dim * num_frames,
    'resnet': 2048,
}

transformer_projection_compact_config = {
    'tactile': 256, 
    'myo': 256, 
    'hand-pose': 256, 
    'joint-rotation': 256,
    'joint-position': 256,
}

position_projection_config = {
    'mlp': signal_latent_dim * num_frames,
    'transformer': signal_latent_dim * num_frames, 
    'mlp-frame': signal_latent_dim, 
}

mlp_projection_compact_config = {
    'tactile': 256, 
    'myo': 256, 
    'hand-pose': 256, 
    'joint-rotation': 256,
    'joint-position': 256,
}

projection_output_dim_config = {
    'open_clip': 1024,
    'clip': 768,
    'resnet': 2048
}

projection_config = {
    'tactile-glove-left': tactile_projection_config,
    'tactile-glove-right': tactile_projection_config,
    'myo-emg-left': position_projection_config, 
    'myo-emg-right': position_projection_config,
    'joint-position': position_projection_config,
    'left-hand-pose': position_projection_config,
    'right-hand-pose': position_projection_config,
    'joint-rotation': position_projection_config, 
    'text': 768, 
    'output_dim': projection_output_dim_config,
}


projection_compact_config = {
    'compact-mlp': mlp_projection_compact_config,
    'compact-transformer': transformer_projection_compact_config,
    'text': 768 * 77}



