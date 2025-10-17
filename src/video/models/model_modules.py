import copy
from collections import OrderedDict
from typing import Tuple, Union, Optional, Any, Callable

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.models as models

from transformers import PretrainedConfig, VivitConfig, VivitModel, VideoMAEConfig, VideoMAEModel

from einops import rearrange
# from human_body_prior.models.vposer_model import NormalDistDecoder, BatchFlatten

from functools import partial
from typing import Callable, List, Optional
import warnings
from timm.models.layers import DropPath, trunc_normal_


from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

from torchsummary import summary





class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=16,
        dropout=0.2
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class ProjectionMLP(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=[512, 768, 768], output_dim=768):
        super().__init__()

        self.projection_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            QuickGELU(),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            # QuickGELU(),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Linear(hidden_dim[1], hidden_dim[2]),
            QuickGELU(),
            nn.Linear(hidden_dim[2], output_dim),
        )

    def forward(self, x):
        projected_feature = self.projection_encoder(x)
        return projected_feature


class TactileEncoder(nn.Module):
    
    def __init__(self, model_config, model_type='heatmap'):
        super().__init__()
        self.model_type = model_type

        if model_type == 'mae':
            configuration = VideoMAEConfig(**model_config['mae'])
            self.tactile_encoder = VideoMAEModel(configuration)        
        elif model_type == 'vivit':
            vivit_model_config = VivitConfig(**model_config['vivit'])
            self.tactile_encoder = VivitModel(vivit_model_config)
        elif model_type == 'conv':
            self.tactile_encoder = ConvVideoEncoder(model_config['conv'])
        elif model_type == 'mvit':
            self.tactile_encoder = MVIT(model_config['mvit'])
        elif model_type == 'neuralfield':
            raise NotImplementedError
        else: 
            self.tactile_encoder = ConvVideoEncoder(model_config['conv'])


    def forward(self, tactile:torch.Tensor) -> torch.Tensor:
        batch_size = tactile.size(0)
        if self.model_type == 'heatmap':
            tactile = tactile.unsqueeze(2)
        
        tactile_embed = self.tactile_encoder(tactile)
        if self.model_type in ['vivit', 'mae']:
            tactile_embed = tactile_embed.last_hidden_state

        return tactile_embed


class PositionEncoder(nn.Module):
    def __init__(self, model_config, model_type='mlp'):
        super().__init__()
        self.model_type = model_type

        if model_type == 'mlp':
            self.positional_encoder = SequentialMLP(**model_config['mlp'])
        elif model_type == 'transformer':
            # self.positional_encoder = SequentialTransformer(**model_config['transformer'])
            raise NotImplementedError
        elif model_type == 'vivit':
            vivit_model_config = VivitConfig(**model_config['vivit'])
            self.positional_encoder = VivitModel(vivit_model_config)
        elif model_type == 'neuralfield':
            raise NotImplementedError
        else: 
            self.positional_encoder = SequentialMLP(**model_config['mlp'])


    def forward(self, input_signal:torch.Tensor) -> torch.Tensor:

        batch_size, length = input_signal.shape[0], input_signal.shape[1]

        if self.model_type == 'transformer':
            input_signal = input_signal.reshape(batch_size, length, -1)

        output_embed = self.positional_encoder(input_signal)

        return output_embed
        



# TODO
class TactileDecoder(nn.Module):
    
    def __init__(self, model_config, model_type='heatmap'):
        super().__init__()
        self.model_type = model_type
   
        if model_type == 'conv':
            self.decoder = SlightlyLargerDeconv(**model_config['conv'])
        elif model_type == 'neuralfield':
            raise NotImplementedError
        else: 
            vivit_model_config = VivitConfig(**model_config['vivit'])
            self.decoder = VivitModel(vivit_model_config)

    def forward(self, tactile:torch.Tensor) -> torch.Tensor:

        b, f  = tactile.shape[0], tactile.shape[1]

        tactile_features = tactile.reshape(b * f, -1)

        out = self.decoder(tactile_features)


        return out

# TODO
class PositionDecoder(nn.Module):
    def __init__(self, model_config, model_type='mlp'):
        super().__init__()
        self.model_type = model_type

        if model_type == 'mlp':
            self.decoder = SequentialMLP(**model_config['mlp'])
        elif model_type == 'neuralfield':
            raise NotImplementedError
        else: 
            self.decoder = SequentialMLP(**model_config['mlp'])


    def forward(self, input_signal:torch.Tensor) -> torch.Tensor:

        batch_size, length = input_signal.shape[0], input_signal.shape[1]

        input_feature = input_signal.reshape(batch_size * length, -1)

        output_embed = self.decoder(input_feature)

        return output_embed

#################################
# make different video encoder
#################################
    
class SmallDeconv(nn.Module):
    def __init__(self, latent_dim=64, in_features=128, out_channels=1, last_layer_sigmoid=True):
        super(SmallDeconv, self).__init__()
        self.last_layer_sigmoid = last_layer_sigmoid
        # self.in_features = 128
        self.in_features = in_features
        self.out_channels = out_channels
        self.fc = nn.Linear(latent_dim, in_features*4)
        self.bn1 = nn.BatchNorm1d(in_features * 4)
        self.layers = nn.Sequential(CNNDecoderBlock(128, 64, stride=2))

        self.conv1 = nn.ConvTranspose2d(64, out_channels, kernel_size=6, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(16)

        # self.conv2 = nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        
        x = self.bn1(torch.relu(self.fc(x)))
        batch_size, bottleneck_size = x.shape[0], x.shape[1]
        x = x.view(batch_size, self.in_features, int(np.sqrt(bottleneck_size//self.in_features)), int(np.sqrt(bottleneck_size//self.in_features)))
        x = self.layers(x)
        x = self.conv1(x)
        if self.last_layer_sigmoid: torch.sigmoid(x)
        return x 




class SlightlyLargerDeconv(nn.Module):
    def __init__(self, in_features=512, out_channels=3, **kwargs):
        super(SlightlyLargerDeconv, self).__init__()
        self.in_features = in_features
        self.out_channels = out_channels

        self.layers = nn.Sequential(CNNDecoderBlock(in_features, 128, stride=2))

        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.ConvTranspose2d(16, out_channels, kernel_size=5, stride=3, padding=2)


    def forward(self, x):
        
        batch_size, frame, bottleneck_size = x.shape[0], x.shape[1], x.shape[2]
        
        x_feat = x.reshape(batch_size * frame, bottleneck_size, 1, 1)

        x_feat = self.layers(x_feat)

        x_feat = torch.relu(self.bn1(self.conv1(x_feat)))

        x_feat = torch.relu(self.bn2(self.conv2(x_feat)))

        x = torch.sigmoid(self.conv3(x_feat))

        return x 



class CNNDecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(CNNDecoderBlock, self).__init__()

        # Settings for settings from different papers
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        return x



class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError


class WaveGanDecoder(nn.Module):
    def __init__(self):
        super(WaveGanDecoder, self).__init__()

        self.Upsample = nn.Upsample(scale_factor=2)
        self.Conv1 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block1 = WaveUnpool(128,"sum").cuda()
        self.Conv2 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block2 = WaveUnpool(128, "sum").cuda()
        self.Conv3 = Conv2dBlock(128, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block3 = WaveUnpool(64, "sum").cuda()
        self.Conv4 = Conv2dBlock(64, 32, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block4 = WaveUnpool(32, "sum").cuda()
        self.Conv5 = Conv2dBlock(32, 3, 5, 1, 2,
                             norm='none',
                             activation='tanh',
                             pad_type='reflect')

    def forward(self, x, skips):
        x1 = self.Upsample(x)
        x2 = self.Conv1(x1)
        LH1, HL1, HH1 = skips['pool4']
        c, h, w = LH1.size()[-3:]
        LH1, HL1, HH1 = LH1.view(8,3,c, h, w).mean(dim=1), HL1.view(8,3,c, h, w).mean(dim=1), HH1.view(8,3,c, h, w).mean(dim=1)
        original1 = skips['conv4_1']
        x_deconv = self.recon_block1(x, LH1, HL1, HH1, original1)
        x2 = x_deconv + x2

        x3 = self.Upsample(x2)
        x4 = self.Conv2(x3)
        LH2, HL2, HH2 = skips['pool3']
        original2 = skips['conv3_1']
        c, h, w = LH2.size()[-3:]
        LH2, HL2, HH2 = LH2.view(8, 3, c, h, w).mean(dim=1), HL2.view(8, 3, c, h, w).mean(dim=1), HH2.view(8, 3, c, h,w).mean(dim=1)
        x_deconv2 = self.recon_block1(x1, LH2, HL2, HH2, original2)

        LH3, HL3, HH3 = skips['pool2']
        c, h, w = skips['conv2_1'].size()[-3:]
#        original3 = skips['conv2_1'].view(8, 3, c, h, w).mean(dim=1)
        c, h, w = LH3.size()[-3:]
        LH3, HL3, HH3 = LH3.view(8, 3, c, h, w).mean(dim=1), HL3.view(8, 3, c, h, w).mean(dim=1), HH3.view(8, 3, c, h,w).mean(dim=1)
        x_deconv4 = self.recon_block1(x3, LH3, HL3, HH3, original2)
        x5 = self.Upsample(x4+x_deconv2)
        x6 = self.Conv3(x5+x_deconv4)

        x7 = self.Upsample(x6)
        x8 = self.Conv4(x7)


        x9 = self.Conv5(x8)

        return x9



class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False, use_cbam=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        # elif norm == 'adain':
        #     self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if norm == 'sn':
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)


    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.cbam:
                x = self.cbam(x)
            if self.activation:
                x = self.activation(x)
        return x




import pytorchvideo.models 

class ConvVideoEncoder(nn.Module):
    def __init__(self, model_config, pooling=False):
        super().__init__()

        self.video_encoder = pytorchvideo.models.x3d.create_x3d(**model_config)
        self.video_encoder = nn.Sequential(*list(self.video_encoder.children())[0][:-1], 
                    nn.Conv3d(192, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
                    nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU())

        if not pooling:
            self.video_encoder = nn.Sequential(*self.video_encoder)
        else:
            self.video_encoder = nn.Sequential(*self.video_encoder, nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0)))


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3, 4)
        encode_feature = self.video_encoder(x)
        encode_feature = encode_feature.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return encode_feature
        


class MVIT(nn.Module):
    def __init__(self, model_config, pooling=False):
        super().__init__()

        self.video_encoder = pytorchvideo.models.create_multiscale_vision_transformers(**model_config)

        if pooling:
            self.video_encoder = nn.Sequential(self.video_encoder, nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0)))


    def forward(self, x):
        # batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3, 4)
        encode_feature = self.video_encoder(x)
        # encode_feature = encode_feature.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return encode_feature
        


class SequentialMLP(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=128, latent_dim=128, n_layers=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            QuickGELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            QuickGELU(),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Linear(hidden_dim, latent_dim),
            # QuickGELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        batch_size, length, = x.shape[0], x.shape[1]
        
        x = x.reshape(batch_size * length, -1)
        embeded_feature = self.encoder(x)
        embeded_feature = embeded_feature.reshape(batch_size, length, -1)
        return embeded_feature
    



# class VPoserEncoder(nn.Module):
#     def __init__(self, input_dim=66, hidden_dim=128, latent_dim=128, type='hand'):
#         super(VPoserEncoder, self).__init__()

#         n_features = input_dim
#         self.encoder_net = nn.Sequential(
#             BatchFlatten(),
#             nn.BatchNorm1d(n_features),
#             nn.Linear(n_features, hidden_dim),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, latent_dim),
#             # NormalDistDecoder(hidden_dim, latent_dim)
#         )

#     def forward(self, x):
#         return self.encoder_net(x)

class BatchLinear(nn.Linear):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        weight = params['weight']
        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))

        if self.bias is not None:
            bias = params.get('bias', None)
            output += bias.unsqueeze(-2)
    
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)






#### CLIP MODULES

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x



class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        net = models.resnet50(pretrained=True)

        net = list(net.children())
        net.pop()
        #self.model_ft =model_ft
        self.net = nn.Sequential(*net)

    def forward(self, x):
        b, t, c, h, w = x.shape

        feat = self.net(rearrange(x, 'b t c h w -> (b t) (c) h w').repeat(1, 3, 1, 1)).squeeze()

        feat = rearrange(feat, '(b t) c -> b t c', b=b, t=t)


        return feat


class ResNetImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        net = models.resnet50(pretrained=True)

        net = list(net.children())
        net.pop()
        #self.model_ft =model_ft
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        
        feat = self.net(x).squeeze()

        return feat


class ResNetUp(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()

        self.upconv6 = upconv(2048, 512, 3, 2)
        self.iconv6 = conv(512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(128, 128, 3, 1)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64, 32, 1, 1)
        self.upconv2 = upconv(32, 16, 3, 2)
        self.iconv2 = conv(16, 4, 1, 1)
        self.iconv1 = conv(4, out_dim, 1, 1)

    def forward(self, x):
        b, t, c = x.shape
        
        x = rearrange(x, 'b t c -> (b t) c 1 1')

        up = self.upconv6(x)
        iconv = self.iconv6(up)
        
        up = self.upconv5(iconv)
        iconv = self.iconv5(up)
        
        up = self.upconv4(iconv)
        iconv = self.iconv4(up)
        
        up = self.upconv3(iconv)
        iconv = self.iconv3(up)
        
        up = self.upconv2(iconv)
        iconv = self.iconv2(up)
        
        iconv = self.iconv1(iconv)
        
        return iconv


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)

class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text




### image bind 

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiheadAttention(nn.MultiheadAttention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return super().forward(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


class ViTAttention(Attention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        assert attn_mask is None
        return super().forward(x)


class BlockWithMasking(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()

        assert not isinstance(
            attn_target, nn.Module
        ), "attn_target should be a Callable. Otherwise attn_target is shared across blocks!"
        self.attn = attn_target()
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm_1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer(dim)
        self.layer_scale_type = layer_scale_type
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask))
            x = x + self.drop_path(self.mlp(self.norm_2(x)))
        else:
            x = (
                x
                + self.drop_path(self.attn(self.norm_1(x), attn_mask))
                * self.layer_scale_gamma1
            )
            x = x + self.drop_path(self.mlp(self.norm_2(x))) * self.layer_scale_gamma2
        return x


_LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        attn_target: Callable,
        embed_dim: int,
        num_blocks: int,
        block: Callable = BlockWithMasking,
        pre_transformer_layer: Optional[Callable] = None,
        post_transformer_layer: Optional[Callable] = None,
        drop_path_rate: float = 0.0,
        drop_path_type: str = "progressive",
        norm_layer: Callable = _LAYER_NORM,
        mlp_ratio: int = 4,
        ffn_dropout_rate: float = 0.0,
        layer_scale_type: Optional[str] = None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value: float = 1e-4,  # from cait; float
        weight_init_style: str = "jax",  # possible values jax or pytorch
    ):
        """
        Simple Transformer with the following features
        1. Supports masked attention
        2. Supports DropPath
        3. Supports LayerScale
        4. Supports Dropout in Attention and FFN
        5. Makes few assumptions about the input except that it is a Tensor
        """
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")

        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_blocks)
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.weight_init_style = weight_init_style
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init_style == "jax":
                # Based on MAE and official Jax ViT implementation
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.weight_init_style == "pytorch":
                # PyTorch ViT uses trunc_normal_
                trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor = None,
        use_checkpoint: bool = False,
        checkpoint_every_n: int = 1,
        checkpoint_blk_ids: Optional[List[int]] = None,
    ):
        """
        Inputs
        - tokens: data of shape N x L x D (or L x N x D depending on the attention implementation)
        - attn: mask of shape L x L

        Output
        - x: data of shape N x L x D (or L x N x D depending on the attention implementation)
        """
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens)
        if use_checkpoint and checkpoint_blk_ids is None:
            checkpoint_blk_ids = [
                blk_id
                for blk_id in range(len(self.blocks))
                if blk_id % checkpoint_every_n == 0
            ]
        if checkpoint_blk_ids:
            checkpoint_blk_ids = set(checkpoint_blk_ids)
        for blk_id, blk in enumerate(self.blocks):
            if use_checkpoint and blk_id in checkpoint_blk_ids:
                tokens = checkpoint.checkpoint(
                    blk, tokens, attn_mask, use_reentrant=False
                )
            else:
                tokens = blk(tokens, attn_mask=attn_mask)
        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens)
        return tokens


