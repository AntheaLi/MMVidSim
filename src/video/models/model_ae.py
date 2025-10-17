import torch
from torch import nn
from einops import rearrange, repeat


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# from exp.action.models.encoders import SignalEncoder, TextEncoder #,SignalDecoder
# from exp.action.models.model_modules import *
# from exp.action.config import projection_config, loss_weight_config

from .modules.i2vgen.autoencoder import I2VAutoEncoder
from .modules.sd.autoencoder import SDAutoencoder

from config import video_model_cfg
from einops import rearrange

class AE(nn.Module):
    def __init__(
        self,
        args, 
        model_type='avdc'
    ):
        super().__init__()

        self.args = args
        self.model_type = model_type
        self.video_model_config = args.video_model_config
        self.dtype = args.video_model_config['dtype']
        self.image_size = args.video_model_config['image_size']
        self.guide_scale = args.video_model_config['guide_scale']
        if self.model_type == 'i2vgen':
            self.ae = I2VAutoEncoder()
        elif self.model_type == 'sd':
            self.ae = SDAutoencoder()
        elif self.model_type in ['avdc', 'flowdiffusion']:
            pass
            # args.video_model_config['avdc']['model'] = Unet64()
        else:
            args.ae = SDAutoencoder()

        self.mse_loss = nn.MSELoss()

    def loss(self, gt, pred):
        return self.mse_loss(pred, gt)
        
    def forward(self, x):
        b, t, h, w, c = x.shape
        pred, z = self.ae(x)

        pred = rearrange(pred, '(b t) c h w -> b t h w c', t=t)
        
        loss = self.loss(x, pred)
        return pred, loss


        