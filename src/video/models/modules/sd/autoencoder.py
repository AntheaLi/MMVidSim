
from torch import nn
from models.modules.sd.diffusionmodules.model import Encoder
from models.modules.sd.autoencoding.temporal_ae import VideoDecoder
import easydict
from exp.video.config import sd_ae_setup
from einops import rearrange

class SDAutoencoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder(**sd_ae_setup['encoder'])
        self.decoder = VideoDecoder(**sd_ae_setup['decoder'])

    def forward(self, x):

        b,f,h,w,c = x.shape
        x = rearrange(x, 'b f h w c -> (b f) c h w')
        z = self.encoder(x)
        y = self.decoder(z, timesteps=f)
        return y, z

    