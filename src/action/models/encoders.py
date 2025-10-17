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


from exp.action.config import text_model_cfg, signal_model_cfg, projection_config, loss_weight_config


class SignalEncoder(nn.Module):
    def __init__(self, 
                 args,
                 signal=None,
                 ):
        super().__init__()
        self.args = args
        if 'tactile' in signal:
            self.model = TactileEncoder(signal_model_cfg['tactile'], model_type=args.tactile_model)
        elif 'joint-position' in signal:
            self.model = PositionEncoder(signal_model_cfg['joint-position'], model_type=args.joint_position_model)
        elif 'hand-pose' in signal:
            self.model = PositionEncoder(signal_model_cfg['hand-pose'], model_type=args.hand_pose_model)
        elif 'joint-rotation' in signal:
            self.model = PositionEncoder(signal_model_cfg['joint-rotation'], model_type=args.joint_rotation_model)
        elif 'myo' in signal:
            self.model = PositionEncoder(signal_model_cfg['myo'], model_type=args.myo_model)
        else:
            self.model = PositionEncoder()

    def forward(self, x):
        return self.model(x)
    


class SignalFrameEncoder(nn.Module):
    def __init__(self, 
                 args,
                 signal=None,
                 ):
        super().__init__()
        self.args = args
        if 'tactile' in signal:
            self.model = ResNet()
        elif 'joint-position' in signal:
            self.model = SequentialMLP(**signal_model_cfg['joint-position'][args.joint_position_model])
        elif 'hand-pose' in signal:
            self.model = SequentialMLP(**signal_model_cfg['hand-pose'][args.hand_pose_model])
        elif 'joint-rotation' in signal:
            self.model = SequentialMLP(**signal_model_cfg['joint-rotation'][args.joint_rotation_model])
        elif 'myo' in signal:
            self.model = SequentialMLP(**signal_model_cfg['myo'][args.myo_model])
        else:
            self.model = SequentialMLP()

    def forward(self, x):

        return self.model(x)
    

class TextEncoder(nn.Module):
    def __init__(self, 
                 model_type='clip',
                 max_length=77,
                 layer=None,
                 ):
        super().__init__()

        self.from_scratch=False
        self.model_type = model_type
        self.max_length = max_length
        self.dtype=torch.float
        self.layer = text_model_cfg['layer'] if layer is None else layer

        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            self.layer_idx = None
        
        if self.model_type == 'clip':
            self.tokenizer = CLIPTokenizer.from_pretrained(text_model_cfg['clip']['version'])
            self.model = CLIPTextModel.from_pretrained(text_model_cfg['clip']['version'])
            self.layer = text_model_cfg['clip']['layer']
            self.layer_idx = text_model_cfg['clip']['layer_idx']
            
        elif model_type == 'open_clip':
            self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained=text_model_cfg['open_clip']['version'])

            del self.model.visual

        elif model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(text_model_cfg['t5']['version'])
            self.model = T5EncoderModel.from_pretrained(text_model_cfg['t5']['version'])

        elif model_type == 'llama':
            # self.tokenizer = transformers.LlamaTokenizerFast(text_model_cfg['llama']['version'])
            # self.model = transformers.LlamaModel.from_pretrained(text_model_cfg['llama']['version'])
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_cfg['llama']['version'])
            self.model = LlamaForCausalLM.from_pretrained(text_model_cfg['llama']['version'])

        else:
            raise NotImplementedError

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_size = len(text[0])
        
        text = text[0]

        device = next(self.model.parameters()).device

        if self.model_type  == 't5':
            batch_encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(device)
            with torch.autocast("cuda", enabled=False):
                outputs = self.model(input_ids=tokens)
            text_features = outputs.last_hidden_state

        elif self.model_type == 'clip':
            batch_encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(device)
        
            outputs = self.model(input_ids=tokens, output_hidden_states=self.layer=="hidden")

            text_features = outputs.last_hidden_state

        elif self.model_type == 'open_clip':
    
            tokens = self.tokenizer(text)
            x = self.model.token_embedding(tokens.to(device))
            x = x + self.model.positional_embedding.to(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.model.transformer(x, attn_mask=self.model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
            text_features = x

        else: 
            raise NotImplementedError

        if self.layer_idx is not None: text_features = text_features[:, self.layer_idx, :].reshape(batch_size, -1)

        return text_features