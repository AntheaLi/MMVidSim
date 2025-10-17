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
from .encoders import SignalEncoder, TextEncoder

# sys.path.append('../')
from config import BaseTransformerCFG as cfg
from config import text_model_cfg, signal_model_cfg, projection_config, loss_weight_config



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
        for sig in self.signals:
            signal_features[sig] = self.signal_encoder[sig](batch[sig]).reshape(batch_size, -1)
            projected_feature[sig] = self.signal_projection[sig](signal_features[sig])


        max_feature = torch.stack([projected_feature[x].unsqueeze(1) for x in self.signals], dim=1).max(1)[0].squeeze()

        max_loss = self.mse_loss(max_feature, encoded_text).mean()

        loss = max_loss

        loss_dict['max loss'] = max_loss


        return loss_dict, loss

        




        