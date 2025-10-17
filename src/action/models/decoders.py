import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers 
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer, AutoTokenizer, LlamaForCausalLM
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
import open_clip
from .model_modules import *

# sys.path.append('../')
from exp.action.config import BaseTransformerCFG as cfg
from exp.action.config import text_model_cfg, signal_model_cfg, signal_model_decoder_cfg, projection_config, loss_weight_config



class SignalDecoder(nn.Module):
    def __init__(self, 
                 args,
                 signal=None,
                 ):
        super().__init__()
        self.args = args
        if 'tactile' in signal:
            self.model = TactileDecoder(signal_model_decoder_cfg['tactile'], model_type=args.tactile_model)
        elif 'joint-position' in signal:
            self.model = PositionDecoder(signal_model_decoder_cfg['joint-position'], model_type=args.joint_position_model)
        elif 'hand-pose' in signal:
            self.model = PositionDecoder(signal_model_decoder_cfg['hand-pose'], model_type=args.hand_pose_model)
        elif 'joint-rotation' in signal:
            self.model = PositionDecoder(signal_model_decoder_cfg['joint-rotation'], model_type=args.joint_rotation_model)
        elif 'myo' in signal:
            self.model = PositionDecoder(signal_model_decoder_cfg['myo'], model_type=args.myo_model)
        else:
            self.model = PositionDecoder()

    def forward(self, x):
        return self.model(x)
    

class SignalFrameDecoder(nn.Module):
    def __init__(self, 
                 args,
                 signal=None,
                 ):
        super().__init__()
        self.args = args
        if 'tactile' in signal:
            self.model = ResNetUp()
            # self.model = SlightlyLargerDeconv(**signal_model_decoder_cfg['tactile']['conv'])
        elif 'joint-position' in signal:
            self.model = SequentialMLP(**signal_model_decoder_cfg['joint-position'][args.joint_position_model])
        elif 'hand-pose' in signal:
            self.model = SequentialMLP(**signal_model_decoder_cfg['hand-pose'][args.hand_pose_model])
        elif 'joint-rotation' in signal:
            self.model = SequentialMLP(**signal_model_decoder_cfg['joint-rotation'][args.joint_rotation_model])
        elif 'myo' in signal:
            self.model = SequentialMLP(**signal_model_decoder_cfg['myo'][args.myo_model])
        else:
            self.model = SequentialMLP()

    def forward(self, x):

        return self.model(x)

