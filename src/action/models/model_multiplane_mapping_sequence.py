import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers 
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
# from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig
# PretrainedConfig, ViTModel, VivitConfig, VivitModel, LlamaModel, DistilBertModel, DistilBertConfig, BertLMHeadModel, EncoderDecoderModel, PreTrainedModel
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
import open_clip
from .model_modules import *

# sys.path.append('../')
from config import BaseTransformerCFG as cfg
from config import text_model_cfg, signal_model_cfg, projection_config, loss_weight_config


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
            self.model = SequentialTransformer()

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, 
                 model_type='clip',
                 max_length=77,
                 ):
        super().__init__()

        self.from_scratch=False
        self.model_type = model_type
        self.max_length = max_length
        self.device=cfg.device
        self.dtype=torch.float
        self.layer = text_model_cfg['layer']

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
            self.tokenizer = transformers.LlamaTokenizerFast(text_model_cfg['llama']['version'])
            self.model = transformers.LlamaModel.from_pretrained(text_model_cfg['llama']['version'])

        else:
            self.from_scratch=True
            self.context_length=cfg.context_length
            self.transformer_width = cfg.transformer_width
            self.vocab_size=cfg.vocab_size
            self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.transformer_width)
            self.positional_embedding = nn.Parameter(torch.empty(cfg.context_length, cfg.transformer_width))
            self.ln_final = LayerNorm(cfg.transformer_width)
            self.model = Transformer(width=cfg.transformer_width, layers=cfg.transformer_layers)

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_size = len(text[0])
        
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
            tokens = batch_encoding["input_ids"].to(self.device)
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
            tokens = batch_encoding["input_ids"].to(self.device)
        
            outputs = self.model(input_ids=tokens, output_hidden_states=self.layer=="hidden")

            text_features = outputs.last_hidden_state

        elif self.model_type == 'open_clip':
    
            tokens = self.tokenizer(text)
            x = self.model.token_embedding(tokens.to(self.device))
            x = x + self.model.positional_embedding.to(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.model.transformer(x, attn_mask=self.model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
            text_features = x

        else: 
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)

            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 

        if self.layer_idx is not None: text_features = text_features[:, self.layer_idx, :].reshape(batch_size, -1)

        return text_features

class Network(nn.Module):
    def __init__(
        self,
        args, 
        temperature=1.0,
        signal_embedding=256, #TODO make this a dictionary of dimensions to be set up in config
        text_embedding=768,
        signals=['tactile-glove-left', 'tactile-glove-right', 'myo-left', 'myo-right', 'joint-rotation', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
    ):
        super().__init__()
        self.signals = signals
        self.pool = args.pool

        self.signal_encoder = nn.ModuleDict()
        self.signal_projection = nn.ModuleDict()

        for sig in signals:
            self.signal_encoder[sig] = SignalEncoder(args, sig)
            self.signal_projection[sig] = ProjectionMLP(input_dim=projection_config[sig][args.signal_model_config[sig]], output_dim=projection_config['output_dim'][args.text_model])

        self.mse_loss = nn.MSELoss()

    def forward(self, batch, encoded_text):

        batch_size = len(batch['label_text'][0])
        # Getting Image and Text Features
        
        signal_features = {}
        projected_feature = {}
        loss_dict = {}
        loss = 0
        for sig in self.signals:
            signal_features[sig] = self.signal_encoder[sig](batch[sig]).reshape(batch_size, -1)
            projected_feature[sig] = self.signal_projection[sig](signal_features[sig])

            loss_sig = self.mse_loss(projected_feature[sig], encoded_text).mean()
            loss += loss_sig * loss_weight_config[sig]
            loss_dict[sig] = loss_sig


        if self.pool == 'avg':
            pool_feature = torch.stack([projected_feature[x].unsqueeze(1) for x in self.signals], dim=1).mean(1).squeeze()
        else:
            pool_feature = torch.stack([projected_feature[x].unsqueeze(1) for x in self.signals], dim=1).max(1)[0].squeeze()

        pool_loss = self.mse_loss(pool_feature, encoded_text).mean()

        loss_dict['pool loss'] = pool_loss

        loss += pool_loss

        return loss_dict, loss

        




        