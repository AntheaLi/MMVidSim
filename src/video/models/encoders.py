# import os, sys
# sys.path.append('./')
# sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers 
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer, AutoTokenizer, LlamaForCausalLM, CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange, repeat
import open_clip

from exp.video.config import text_model_cfg, image_model_cfg
from exp.video.action_config import signal_model_cfg, signal_model_decoder_cfg, projection_config, loss_weight_config

# from exp.action.models.model_modules import SequentialMLP, ModifiedResNet, ResNet, ResNetUp, TactileEncoder, PositionEncoder, TactileDecoder, PositionDecoder, SlightlyLargerDeconv
from exp.video.models.model_modules import SequentialMLP, ModifiedResNet, ResNet, ResNetUp, TactileEncoder, PositionEncoder, TactileDecoder, PositionDecoder, SlightlyLargerDeconv, ResNetImageEncoder




class ImageEncoder(nn.Module):
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
        self.layer = image_model_cfg['layer'] if layer is None else layer

        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            self.layer_idx = None

        if self.model_type == 'clip':
            self.preprocess = CLIPImageProcessor.from_pretrained(image_model_cfg['clip']['version'])
            self.model = CLIPVisionModel.from_pretrained(image_model_cfg['clip']['version'])
            self.layer = image_model_cfg['clip']['layer']
            self.layer_idx = image_model_cfg['clip']['layer_idx']
            
        elif model_type == 'open_clip':
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(image_model_cfg['open_clip']['arch'], pretrained=image_model_cfg['open_clip']['version'])

            
        elif 'resnet' in model_type :
            self.model = ResNetImageEncoder()

        else:
            raise NotImplementedError

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img):
        batch_size = img.shape[0]
        
        if 'resnet' in self.model_type:
            image_features = self.model(img)

        elif self.model_type == 'clip':
            img = rearrange(img, 'b h w c -> b c h w')
            outputs = self.model(img)
            image_features = outputs.last_hidden_state
        
        elif self.model_type == 'open_clip':
            img = rearrange(img, 'b h w c -> b c h w')
            image_features = self.model.encode_image(img)

        else: 
            raise NotImplementedError
        
        # if self.layer_idx is not None: text_features = text_features[:, self.layer_idx, :].reshape(batch_size, -1)

        return image_features


EPS=1e-6
class GlobalActivationLeakyReLULinear(nn.Module):
    def __init__(self, d_input_dim=1024, p_input_dim=2048,  output_dim=1024, fusion='late', merge_local='max', 
                signals=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
                 ):
        super().__init__()
        self.signals = signals
        self.signal_direction = nn.Linear(d_input_dim, output_dim, bias=False)
        self.hyperplane_direction = nn.Linear(p_input_dim, output_dim, bias=False)
        self.negative_slope = 0.0
        self.merge_local = merge_local
        if self.merge_local == 'softmax': 
            self.confidence_layer_d = nn.Linear(d_input_dim, d_input_dim)
            
          
          
    def merge(self, x):
        expand_dim=2
        x_keys = list(x.keys())
        if self.merge_local == 'max':
            x_merge = torch.stack([x[s].unsqueeze(expand_dim) for s in x_keys], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
        elif self.merge_local == 'softmax':
            logits = torch.stack([x[s].unsqueeze(expand_dim) for s in x_keys], dim=expand_dim)
            mask = torch.softmax(torch.relu(self.confidence_layer_d(logits)), dim=expand_dim)
            x_merge = torch.sum(mask * logits, dim=expand_dim).squeeze()
        elif self.merge_local == 'sum':
            x_merge = torch.stack([x[s].unsqueeze(expand_dim) for s in x_keys], dim=expand_dim).sum(expand_dim).squeeze() # B x T x D
        elif self.merge_local == 'cat':
            x_merge = torch.cat([x[s] for s in x_keys], dim=-1) # B x T x D
        elif self.merge_local == 'cross':
            x_shape = x[x.keys()[0]].shape
            x_t = torch.cat([x[s] for s in x_keys], dim=-1)
            x_t = x_t.reshape(-1, 1)
            x_atten = torch.cross(x_t, x_t.T)
            x_atten_pact = x_atten.sum(-1).reshape(*x_shape)
            return x_atten_pact
        return x_merge
        

    def forward(self, x, p, x_res=None, p_res=None, num_frames=8):
        

        expand_dim = 1
        if isinstance(x, dict):
            # x = torch.stack([x[s].unsqueeze(expand_dim) for s in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
            x = self.merge(x)

        if x_res is not None:
            # if isinstance(x_res, dict): x_res = torch.stack([x_res[s].unsqueeze(expand_dim) for s in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
            if isinstance(x_res, dict): 
                x_res = self.merge(x_res)
            x = x + x_res
                    
        d = self.signal_direction(x)
        
        if p_res is not None:
            p = torch.cat([p, p_res], dim=-1)
        
        p = self.hyperplane_direction(p)

        # feature meshing for stepwise excitation  (action-neuron) B x T x D
        # use temporal global feature as activation
        p = repeat(p, 'b d -> b t d', t = num_frames) 
        
        # do this over T
        dotprod = (d*p).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        p_norm_sq = (p*p).sum(2, keepdims=True)
        signal_activation = self.negative_slope * d + (1-self.negative_slope) * (mask*d + (1-mask)*(d-(dotprod/(p_norm_sq+EPS))*p))

        return signal_activation, p[:, 0, :]
    


    
EPS=1e-6
class LocalActivationLeakyReLULinear(nn.Module):
    def __init__(self, d_input_dim=1024, p_input_dim=2048,  output_dim=1024, fusion='late',
                signals=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
                 ):
        super().__init__()
        self.signals = signals
        self.signal_direction = nn.ModuleDict()
        
        for sig in signals:
            self.signal_direction[sig] = nn.Linear(d_input_dim, output_dim, bias=False)
        self.hyperplane_direction = nn.Linear(p_input_dim, output_dim, bias=False)
        
        self.negative_slope = 0.0
    
    
          
    def forward(self, x, p, x_res=None, p_res=None):
        
        batch_size, num_frames  = x[self.signals[0]].shape[0], x[self.signals[0]].shape[1]
        x_keys = list(x.keys())
        if p_res is not None:
            p = torch.cat([p, p_res], dim=-1)
        
        p = self.hyperplane_direction(p)

        # feature meshing for stepwise excitation  (action-neuron) B x T x D
        # use temporal global feature as activation
        p = repeat(p, 'b d -> b t d', t = num_frames) 

        expand_dim = 1
        x_global = torch.stack([x[s].unsqueeze(expand_dim) for s in x_keys], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
            
        action_feature = {}
        for sig in x_keys:
            x[sig] = torch.cat([x_global, x[sig]], dim=-1)
            if x_res is not None: x[sig] = x[sig] + x_res[sig]
            d = self.signal_direction[sig](x[sig])
            
            # do this over T
            dotprod = (d*p).sum(2, keepdims=True)
            mask = (dotprod >= 0).float()
            p_norm_sq = (p*p).sum(2, keepdims=True)
            action_feature[sig] = self.negative_slope * d + (1-self.negative_slope) * (mask*d + (1-mask)*(d-(dotprod/(p_norm_sq+EPS))*p))

        return action_feature, p[:, 0, :]

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
        out = self.model(x)
        return out
        # return self.model(x)
    

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
        device = next(self.model.parameters()).device
        text = text[0]

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