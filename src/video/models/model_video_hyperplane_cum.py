import torch
from torch import nn
from einops import rearrange, repeat


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import I2VGenXLPipeline, DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
from diffusers.schedulers.scheduling_ddim import rescale_zero_terminal_snr

from exp.video.models.encoders import SignalEncoder, TextEncoder, SignalFrameEncoder, SignalDecoder, SignalFrameDecoder, GlobalActivationLeakyReLULinear, LocalActivationLeakyReLULinear
from exp.action.models.model_modules import *
from exp.video.action_config import projection_config, loss_weight_config
from exp.video.config import text_model_cfg, image_model_cfg

from .modules.i2vgen.autoencoder import AutoencoderKL
from .modules.i2vgen.unet import UNetSD_I2VGen
from .modules.i2vgen.unet.unet_ours import UNetSD_Our
from .modules.i2vgen.unet.unet_ours_lcd import UNetSD_Our_LCD
from .modules.i2vgen.unet.unet_3d_huggingface import I2VGenUNetTiny
from .modules.i2vgen.diffusion.diffusion_ddim import DiffusionDDIM
from .modules.flowdiffusion.goal_diffusion import GoalGaussianDiffusion
from .modules.flowdiffusion.goal_diffusion_ours_lcd import GoalGaussianDiffusionLCD
from .modules.flowdiffusion.goal_diffusion_i2v import GoalGaussianDiffusionI2V
from .modules.flowdiffusion.unet import Unet64
# from .modules.sd.diffusionmodules.video_model import VideoUNet
from config import video_model_cfg

EPS = 1e-6

class VideoDiffusion(nn.Module):
    def __init__(
        self,
        args, 
        use_text=False, 
        model_type='avdc'
    ):
        super().__init__()

        self.args = args
        self.model_type = model_type
        self.video_model_config = args.video_model_config
        self.dtype = args.video_model_config['dtype']
        self.image_size = args.video_model_config['image_size']
        self.guide_scale = args.video_model_config['guide_scale']
        self.num_frames = args.video_model_config['num_frames']
        self.use_text = use_text
        if self.model_type == 'i2vgen':
            self.unet = UNetSD_I2VGen(**args.video_model_config['i2vgen']['unet'])
            self.pipeline = DiffusionDDIM(self.unet, **args.video_model_config['i2vgen']['diffusion'])    
        elif self.model_type == 'i2vgen-avdc':
            self.unet = UNetSD_I2VGen(**args.video_model_config['i2vgen']['unet'])
            self.pipeline = GoalGaussianDiffusionLCD(self.unet, **args.video_model_config['i2vgen-avdc'])    
        elif self.model_type == 'ours':
            self.unet = UNetSD_Our(**args.video_model_config['our']['unet'])
            self.pipeline = GoalGaussianDiffusionLCD(self.unet, **args.video_model_config['i2vgen-avdc'])    
        elif self.model_type == 'ours-lcd':
            self.unet = UNetSD_Our_LCD(**args.video_model_config['our']['unet-lcd'])
            self.pipeline = GoalGaussianDiffusionLCD(self.unet, **args.video_model_config['i2vgen-avdc'])    
        elif self.model_type == 'ours-i2v':
            self.unet = I2VGenUNetTiny()
            self.pipeline = GoalGaussianDiffusionI2V(self.unet, **args.video_model_config['i2vgen-avdc'])
        elif self.model_type == 'i2vgen-xl':
            self.pipeline = I2VGenXLPipeline.from_pretrained(args.video_model_config['i2vgen']['version'], torch_dtype=torch.float16, variant="fp16").to("cuda")        
        elif self.model_type == 'sd':
            self.unet = VideoUNet(**args.video_model_config['sd']['unet'])
            self.pipeline = GoalGaussianDiffusionLCD(self.unet, **args.video_model_config['sd-avdc'])    
        elif self.model_type == 'sd-avdc':
            self.unet = VideoUNet(**args.video_model_config['sd']['unet'])
            self.pipeline = GoalGaussianDiffusionLCD(self.unet, **args.video_model_config['sd-avdc'])    
        elif self.model_type in ['avdc', 'flowdiffusion']:
            args.video_model_config['avdc']['model'] = Unet64()
            self.pipeline = GoalGaussianDiffusion(**args.video_model_config['avdc'])    
        else:
            args.video_model_config['avdc']['model'] = Unet64()
            self.pipeline = GoalGaussianDiffusion(**args.video_model_config['avdc'])

    def sample(self, image, action_data, batch_size=1, image_feat=None, negative_task_emb=None):
        if self.model_type == 'i2vgen':
            image = image[:,:, 0:1].detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, negative_task_emb=negative_task_emb, guide_scale=self.guide_scale)
        elif self.model_type == 'i2vgen-avdc':
            image = image[:,:, 0:1].detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, guide_scale=self.guide_scale)
        elif self.model_type == 'ours-lcd':
            if self.use_text: action_data = repeat(action_data, 'b c -> b f c', f=self.num_frames)
            image = image[:,:, 0:1].detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, image_feat=image_feat, guide_scale=self.guide_scale)
        elif self.model_type == 'ours-i2v':
            if self.use_text: action_data = repeat(action_data, 'b c -> b f c', f=self.num_frames)
            image = image[:,:, 0:1].detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, image_feat=image_feat, guide_scale=self.guide_scale)
        else:
            image = image.detach().clone()
        return self.pipeline.sample(image, action_data, batch_size=batch_size, guide_scale=self.guide_scale)

        
    def forward(self, x, action_embedding, image_feat=None, text_feature=None, negative_task_emb=None):
        x = x.permute(0, 1, 4, 2, 3)
        b, t, c, h, w = x.shape
        image = x[:,0:1].squeeze().detach().clone()
        target = x[:, 1:]
    
        if self.model_type == 'i2vgen':
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            # TODO: do not include fps at the moment
            # fps_tensor =  torch.tensor([self.args.video_model_config['i2vgen']['diffusion']['fps']] * b, dtype=torch.long, device=target.device)
            loss, model_pred = self.pipeline(target, image, action_embedding, negative_task_emb=negative_task_emb)
        
        elif self.model_type == 'i2vgen-avdc':
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            loss, model_pred = self.pipeline(target, image, action_embedding)

        elif self.model_type == 'ours': #TODO
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            loss, model_pred = self.pipeline(target, image, action_embedding)

        elif self.model_type == 'ours-lcd': #TODO
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            if self.use_text: action_embedding = repeat(action_embedding, 'b c -> b f c', f=t-1)
            loss, model_pred = self.pipeline(target, image, action_embedding, image_feat=image_feat)

        elif self.model_type == 'ours-i2v': #TODO
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            if self.use_text: action_embedding = repeat(action_embedding, 'b c -> b f c', f=t-1)
            loss, model_pred = self.pipeline(target, image, action_embedding, image_feat=image_feat)

        elif self.model_type == 'sd-avdc':
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            loss, model_pred = self.pipeline(target, image, action_embedding)

        elif self.model_type == 'i2vgen-xl':
            generator = torch.manual_seed(8888)
            model_pred = self.pipeline(prompt=x['text_label'],image=image,generator=generator).frames[0]
            loss = F.mse_loss(model_pred, target, reduction="mean")
            
        elif self.model_type in ['avdc', 'flowdiffusion']:
            
            target = target.reshape(b, (t-1) * c, h, w)
            loss, model_pred = self.pipeline( target, image, action_embedding)
        
        else:       
            loss, model_pred = self.pipeline( target, image, action_embedding)

        return loss, model_pred



class GlobalHyperplane(nn.Module):
    def __init__(
        self,
        args, 
        temperature=1.0,
        signal_embedding=256, #TODO make this a dictionary of dimensions to be set up in config
        text_embedding=768,
        signals=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
    ):
        super().__init__()
        self.signals = signals
        self.pool = args.pool

        self.signal_encoder = nn.ModuleDict()
        self.signal_projection = nn.ModuleDict()
        self.signal_decoder = nn.ModuleDict()
        self.merge_local = args.merge_local

        for sig in signals:
            self.signal_encoder[sig] = SignalFrameEncoder(args, sig)
            self.signal_projection[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_decoder[sig] = SignalFrameDecoder(args, sig)

        if self.merge_local == 'softmax': 
            self.confidence = nn.Linear(projection_config['output_dim'][args.text_model], projection_config['output_dim'][args.text_model])

        self.signal_direction = nn.Linear(projection_config['output_dim'][args.text_model], projection_config['output_dim'][args.text_model], bias=False)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.negative_slope = 0.0
        self.merge_feat = 'activation'


    def merge(self, x):
        expand_dim=2
        if self.merge_local == 'max':
            x_merge = torch.stack([x[s].unsqueeze(expand_dim) for s in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
        elif self.merge_local == 'softmax':
            logits = torch.stack([x[s].unsqueeze(expand_dim) for s in self.signals], dim=expand_dim)
            mask = torch.softmax(torch.relu(self.confidence_layer(logits)), dim=expand_dim)
            x_merge = torch.sum(mask * logits, dim=expand_dim)
        elif self.merge_local == 'sum':
            x_merge = torch.stack([x[s].unsqueeze(expand_dim) for s in self.signals], dim=expand_dim).sum(expand_dim).squeeze() # B x T x D
        elif self.merge_local == 'cat':
            x_merge = torch.cat([x[s] for s in self.signals], dim=-1) # B x T x D
        elif self.merge_local == 'cross':
            x_shape = x[x.keys()[0]].shape
            x_t = torch.cat([x[s] for s in self.signals], dim=-1)
            x_t = x_t.reshape(-1, 1)
            x_atten = torch.cross(x_t, x_t.T)
            x_atten_pact = x_atten.sum(-1).reshape(*x_shape)
            return x_atten_pact
        return x_merge
        

    def forward(self, batch):

        batch_size, num_frames  = batch[self.signals[0]].shape[0], batch[self.signals[0]].shape[1]
        # Getting Image and Text Features
        
        temporal_features = {}
        signal_features = {}
        projected_feature = {}
        signal_reconstruction = {}
        
        loss = 0
        for sig in self.signals:
            signal_features[sig] = self.signal_encoder[sig](batch[sig])
            recon_sig = self.signal_decoder[sig](signal_features[sig])
            if recon_sig.shape != batch[sig].shape: signal_reconstruction[sig] = recon_sig.reshape(batch_size, num_frames, *batch[sig].shape[2:])
            else: signal_reconstruction[sig] = recon_sig
            projected_feat = self.signal_projection[sig](signal_features[sig].reshape(batch_size * num_frames, -1)) # B x T x D
            projected_feature[sig] = rearrange(projected_feat, '(b t) d -> b t d', b=batch_size)

            # reconstruction loss
            # loss_sig = self.mse_loss(signal_reconstruction[sig], batch[sig]).mean()
            # loss += loss_sig
            # loss_dict[sig] = loss_sig


        # pooling over signals max over S [projected feature B x T x S x D -> B x T x D]
        expand_dim = 1
        
        # action_feature = torch.stack([projected_feature[x].unsqueeze(expand_dim) for x in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
        action_feature = self.merge(projected_feature)
        
        # pooling across temporal dimension
        temporal_global_feature = action_feature.mean(1).squeeze() # pooling across temporal dimension 
        temporal_global_feature = self.signal_direction(temporal_global_feature)
        # alignment_loss = - self.cos_loss(temporal_global_feature, text_feature).mean()
        # loss += alignment_loss
        # loss_dict['align'] = alignment_loss

        # feature meshing for stepwise excitation  (action-neuron) B x T x D
        # use temporal global feature as activation
        p = repeat(temporal_global_feature, 'b d -> b t d', t = num_frames) 
        
        # do this over T
        d = action_feature
        dotprod = (d*p).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        p_norm_sq = (p*p).sum(2, keepdims=True)
        signal_activation = self.negative_slope * d + (1-self.negative_slope) * (mask*d + (1-mask)*(d-(dotprod/(p_norm_sq+EPS))*p))

        # conditional feature meshing 
        if self.merge_feat == 'concat':
            x_cond = torch.cat([p, signal_activation], dim=-1)
        elif self.merge_feat == 'add':
            x_cond = p + signal_activation 
        else:
            x_cond = signal_activation 

        out_dict = {
            'cond_feature': x_cond, 
            'p': temporal_global_feature,
            'a': signal_activation, 
            'recon': signal_reconstruction}

        return out_dict
    


EPS = 1e-6    
class IndividualHyperPlane(nn.Module):
    def __init__(
        self,
        args, 
        temperature=1.0,
        signal_embedding=256, #TODO make this a dictionary of dimensions to be set up in config
        text_embedding=768,
        signals=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
    ):
        super().__init__()
        self.signals = signals
        self.pool = args.pool
        self.merge_local = args.merge_local

        self.signal_encoder = nn.ModuleDict()
        self.signal_projection = nn.ModuleDict()
        self.signal_direction = nn.ModuleDict()
        self.signal_decoder = nn.ModuleDict()

        for sig in signals:
            self.signal_encoder[sig] = SignalFrameEncoder(args, sig)
            self.signal_projection[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_direction[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_decoder[sig] = SignalFrameDecoder(args, sig)


        if self.merge_local == 'softmax':
            self.confidence_layer = nn.Linear(projection_config['output_dim'][args.text_model], projection_config['output_dim'][args.text_model])

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.negative_slope = 0.0
        self.merge_feat = 'activation'


    def merge(self, x):
        expand_dim=2
        if self.merge_local == 'max':
            x_merge = torch.stack([x[s].unsqueeze(expand_dim) for s in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
        elif self.merge_local == 'softmax':
            logits = torch.stack([x[s].unsqueeze(expand_dim) for s in self.signals], dim=expand_dim)
            mask = torch.softmax(torch.relu(self.confidence_layer(logits)), dim=expand_dim)
            x_merge = torch.sum(mask * logits, dim=expand_dim)
        elif self.merge_local == 'sum':
            x_merge = torch.stack([x[s].unsqueeze(expand_dim) for s in self.signals], dim=expand_dim).sum(expand_dim).squeeze() # B x T x D
        elif self.merge_local == 'cat':
            x_merge = torch.cat([x[s] for s in self.signals], dim=-1) # B x T x D
        elif self.merge_local == 'cross':
            x_shape = x[x.keys()[0]].shape
            x_t = torch.cat([x[s] for s in self.signals], dim=-1)
            x_t = x_t.reshape(-1, 1)
            x_atten = torch.cross(x_t, x_t.T)
            x_atten_pact = x_atten.sum(-1).reshape(*x_shape)
            return x_atten_pact
        return x_merge
    


    def forward(self, batch):

        batch_size, num_frames  = batch[self.signals[0]].shape[0], batch[self.signals[0]].shape[1]
        # Getting Image and Text Features
        
        temporal_features = {}
        signal_features = {}
        projected_features = {}
        signal_reconstruction = {}
        signal_activation = {}
        loss_dict = {}
        loss = 0
        for sig in self.signals:
            signal_features[sig] = self.signal_encoder[sig](batch[sig])
            recon_sig = self.signal_decoder[sig](signal_features[sig])
            if recon_sig.shape != batch[sig].shape: signal_reconstruction[sig] = recon_sig.reshape(batch_size, num_frames, *batch[sig].shape[2:])
            else: signal_reconstruction[sig] = recon_sig

            project_feature = self.signal_projection[sig](signal_features[sig].reshape(batch_size * num_frames, -1)) # B x T x D
            d = self.signal_direction[sig](signal_features[sig].reshape(batch_size * num_frames, -1)) # B x T x D
            d = rearrange(d, '(b t) d -> b t d', t=num_frames)

            project_feature = rearrange(project_feature, '(b t) d -> b t d', t=num_frames).mean(1).squeeze() # pooling across temporal dimension B x D
            
            if project_feature.ndim == 1: project_feature = project_feature.unsqueeze(0)
            p = repeat(project_feature, 'b d -> b t d', t = num_frames) 
            projected_features[sig] = p

            # do this over T    
            dotprod = (d*p).sum(2, keepdims=True)
            mask = (dotprod >= 0).float()
            p_norm_sq = (p*p).sum(2, keepdims=True)
            signal_activation[sig] = self.negative_slope * d + (1-self.negative_slope) * (mask*d + (1-mask)*(d-(dotprod/(p_norm_sq+EPS))*p))


        # align global plane -> max over plane 
        expand_dim = 2
        temporal_global_feature = torch.stack([projected_features[x].unsqueeze(expand_dim) for x in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D

        # conditional feature 
        # signal_activation_max = torch.stack([signal_activation[x].unsqueeze(expand_dim) for x in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D        
        signal_activation_max = self.merge(signal_activation)

        # conditional feature meshing 
        if self.merge_feat == 'concat':
            x_cond = torch.cat([temporal_global_feature, signal_activation_max], dim=-1)
        elif self.merge_feat == 'add':
            x_cond = temporal_global_feature + signal_activation_max 
        else:
            x_cond = signal_activation_max 



        out_dict = {
            'cond_feature': x_cond, 
            'p': temporal_global_feature[:, 0, :].squeeze(),
            'a': signal_activation_max,
            'recon': signal_reconstruction}



        return out_dict
  


EPS = 1e-6    
class GlobalHyperplaneMetric(nn.Module):
    ''' Global Hyperplane activation with one layer
    '''
    def __init__(
        self,
        args, 
        temperature=1.0,
        signal_embedding=256, #TODO make this a dictionary of dimensions to be set up in config
        text_embedding=768,
        signals=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
    ):
        super().__init__()
        self.signals = signals
        self.pool = args.pool

        self.signal_encoder = nn.ModuleDict()
        self.signal_projection = nn.ModuleDict()
        self.signal_decoder = nn.ModuleDict()
        
        self.merge_local = args.merge_local
        
        if self.merge_local == 'softmax': 
            self.confidence_layer = nn.Linear(projection_config['output_dim'][args.text_model], projection_config['output_dim'][args.text_model])
        

        for sig in signals:
            self.signal_encoder[sig] = SignalFrameEncoder(args, sig)
            self.signal_projection[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_decoder[sig] = SignalFrameDecoder(args, sig)


        img_feature_dim = image_model_cfg[args.image_model]['dimension']
        self.signal_direction = nn.Sequential(nn.Linear(img_feature_dim, projection_config['output_dim'][args.text_model]), 
                                            nn.LeakyReLU(), 
                                            nn.Linear(projection_config['output_dim'][args.text_model], projection_config['output_dim'][args.text_model], bias=False))
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.negative_slope = 0.0
        self.merge_feat = 'activation'
        self.cos_loss_dir = 1.0
        self.cos_loss_magnitude = 1.0

    def merge(self, x):
        expand_dim=2
        x_keys = list(x.keys())
        if self.merge_local == 'max':
            x_merge = torch.stack([x[s].unsqueeze(expand_dim) for s in x_keys], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
        elif self.merge_local == 'softmax':
            logits = torch.stack([x[s].unsqueeze(expand_dim) for s in x_keys], dim=expand_dim)
            mask = torch.softmax(torch.relu(self.confidence_layer(logits)), dim=expand_dim)
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

    def forward(self, batch):

        batch_size, num_frames  = batch[self.signals[0]].shape[0], batch[self.signals[0]].shape[1]
        # Getting Image and Text Features
        
        temporal_features = {}
        signal_features = {}
        projected_feature = {}
        signal_reconstruction = {}
        for sig in self.signals:
            signal_features[sig] = self.signal_encoder[sig](batch[sig])
            recon_sig = self.signal_decoder[sig](signal_features[sig])
            if recon_sig.shape != batch[sig].shape: signal_reconstruction[sig] = recon_sig.reshape(batch_size, num_frames, *batch[sig].shape[2:])
            else: signal_reconstruction[sig] = recon_sig
            projected_feat = self.signal_projection[sig](signal_features[sig].reshape(batch_size * num_frames, -1)) # B x T x D
            projected_feature[sig] = rearrange(projected_feat, '(b t) d -> b t d', b=batch_size)

        if 'text' in batch and len(self.signals) >= 7:
            # print('compare shape of different ', batch['text'].shape, projected_feature[self.signals[0]].shape)
            projected_feature['text'] = repeat(batch['text'], 'b d -> b t d', t=num_frames)
            # print(projected_feature['text'].shape)

        # pooling over signals max over S [projected feature B x T x S x D -> B x T x D]
        # expand_dim = 1
        # action_feature = torch.stack([projected_feature[x].unsqueeze(expand_dim) for x in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
        action_feature = self.merge(projected_feature)

        # get plane normal direction
        p = self.signal_direction(batch['context'])

        # feature meshing for stepwise excitation  (action-neuron) B x T x D
        # use temporal global feature as activation
        p = repeat(p, 'b d -> b t d', t = num_frames) 
        
        # do this over T
        d = action_feature
        dotprod = (d*p).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        p_norm_sq = (p*p).sum(2, keepdims=True)
        signal_activation = self.negative_slope * d + (1-self.negative_slope) * (mask*d + (1-mask)*(d-(dotprod/(p_norm_sq+EPS))*p))

        # conditional feature meshing 
        if self.merge_feat == 'concat':
            x_cond = torch.cat([p, signal_activation], dim=-1)
        elif self.merge_feat == 'add':
            x_cond = p + signal_activation 
        else:
            x_cond = signal_activation 

        out_dict = {
            'cond_feature': x_cond, 
            'p': p[:, 0, :].squeeze(),
            'a': signal_activation,
            'recon': signal_reconstruction}
        
        return out_dict




EPS = 1e-6    
class GlobalHyperplaneMetricLayers(nn.Module):
    def __init__(
        self,
        args, 
        temperature=1.0,
        signal_embedding=256, #TODO make this a dictionary of dimensions to be set up in config
        text_embedding=768,
        num_layers=5, 
        signals=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
    ):
        super().__init__()
        self.signals = signals
        self.pool = args.pool

        self.signal_encoder = nn.ModuleDict()
        self.signal_projection = nn.ModuleDict()
        self.signal_decoder = nn.ModuleDict()
        

        for sig in signals:
            self.signal_encoder[sig] = SignalFrameEncoder(args, sig)
            self.signal_projection[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_decoder[sig] = SignalFrameDecoder(args, sig)

        img_feature_dim = image_model_cfg[args.image_model]['dimension']

        output_dim = projection_config['output_dim'][args.text_model]
        x_indim = projection_config['output_dim'][args.text_model]
        p_indim = img_feature_dim + projection_config['output_dim'][args.text_model]
            
        self.linear1 = GlobalActivationLeakyReLULinear(d_input_dim=x_indim, p_input_dim=img_feature_dim, output_dim=output_dim, merge_local=args.merge_local, signals=signals)
        self.linear2 = GlobalActivationLeakyReLULinear(d_input_dim=x_indim, p_input_dim=p_indim, output_dim=output_dim, merge_local=args.merge_local, signals=signals)
        self.linear3 = GlobalActivationLeakyReLULinear(d_input_dim=x_indim, p_input_dim=p_indim, output_dim=output_dim, merge_local=args.merge_local, signals=signals)
        

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.negative_slope = 0.0
        self.merge_feat = 'activation'
        self.cos_loss_dir = 1.0
        self.cos_loss_magnitude = 1.0


    def forward(self, batch):

        batch_size, num_frames  = batch[self.signals[0]].shape[0], batch[self.signals[0]].shape[1]
        # Getting Image and Text Features
        
        temporal_features = {}
        signal_features = {}
        projected_feature = {}
        signal_reconstruction = {}
        for sig in self.signals:
            signal_features[sig] = self.signal_encoder[sig](batch[sig])
            recon_sig = self.signal_decoder[sig](signal_features[sig])
            if recon_sig.shape != batch[sig].shape: signal_reconstruction[sig] = recon_sig.reshape(batch_size, num_frames, *batch[sig].shape[2:])
            else: signal_reconstruction[sig] = recon_sig
            projected_feat = self.signal_projection[sig](signal_features[sig].reshape(batch_size * num_frames, -1)) # B x T x D
            projected_feature[sig] = rearrange(projected_feat, '(b t) d -> b t d', b=batch_size)

        if 'text' in batch and len(self.signals) >= 7:
            # print('compare shape of different ', batch['text'].shape, projected_feature[self.signals[0]].shape)
            projected_feature['text'] = repeat(batch['text'], 'b d -> b t d', t=num_frames)


        # pooling over signals max over S [projected feature B x T x S x D -> B x T x D]
        activation, hyperplane = self.linear1(projected_feature, batch['context'], x_res=None, p_res=None)
        activation, hyperplane = self.linear2(activation, hyperplane, x_res=projected_feature, p_res=batch['context'])
        activation, hyperplane = self.linear3(activation, hyperplane, x_res=projected_feature, p_res=batch['context'])


        # activation = torch.cumsum(activation, dim=1)

        # conditional feature meshing 
        if self.merge_feat == 'concat':
            x_cond = torch.cat([hyperplane, activation], dim=-1)
        elif self.merge_feat == 'add':
            x_cond = hyperplane + activation 
        else:
            x_cond = activation 

        out_dict = {
            'cond_feature': x_cond, 
            'p': hyperplane,
            'a': activation,
            'recon': signal_reconstruction}
        
        return out_dict




EPS = 1e-6    
class LocalHyperplaneMetricLayers(nn.Module):
    def __init__(
        self,
        args, 
        temperature=1.0,
        signal_embedding=256, #TODO make this a dictionary of dimensions to be set up in config
        text_embedding=768,
        signals=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
    ):
        super().__init__()
        self.signals = signals
        self.merge_local = args.merge_local

        self.signal_encoder = nn.ModuleDict()
        self.signal_projection = nn.ModuleDict()
        self.signal_decoder = nn.ModuleDict()
        

        for sig in signals:
            self.signal_encoder[sig] = SignalFrameEncoder(args, sig)
            self.signal_projection[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_decoder[sig] = SignalFrameDecoder(args, sig)
            
        if self.merge_local == 'softmax':
            self.confidence_layer = nn.Linear(projection_config['output_dim'][args.text_model],  projection_config['output_dim'][args.text_model])


        img_feature_dim = image_model_cfg[args.image_model]['dimension']

        output_dim = projection_config['output_dim'][args.text_model]
        x_indim = 2 * projection_config['output_dim'][args.text_model]
        p_indim = img_feature_dim + projection_config['output_dim'][args.text_model]
            
        self.linear1 = LocalActivationLeakyReLULinear(d_input_dim=x_indim, p_input_dim=img_feature_dim, output_dim=output_dim, signals=signals)
        self.linear2 = LocalActivationLeakyReLULinear(d_input_dim=x_indim, p_input_dim=p_indim, output_dim=output_dim, signals=signals)
        self.linear3 = LocalActivationLeakyReLULinear(d_input_dim=x_indim, p_input_dim=p_indim, output_dim=output_dim, signals=signals)
        
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.negative_slope = 0.0
        self.merge_feat = 'activation'
        self.cos_loss_dir = 1.0
        self.cos_loss_magnitude = 1.0

    def merge(self, x):
        expand_dim=2
        x_keys = list(x.keys())
        if self.merge_local == 'max':
            x_merge = torch.stack([x[s].unsqueeze(expand_dim) for s in x_keys], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D
        elif self.merge_local == 'softmax':
            logits = torch.stack([x[s].unsqueeze(expand_dim) for s in x_keys], dim=expand_dim)
            mask = torch.softmax(torch.relu(self.confidence_layer(logits)), dim=expand_dim)
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
        
        

    def forward(self, batch):

        batch_size, num_frames  = batch[self.signals[0]].shape[0], batch[self.signals[0]].shape[1]
        # Getting Image and Text Features
        
        temporal_features = {}
        signal_features = {}
        projected_feature = {}
        signal_reconstruction = {}
        for sig in self.signals:
            signal_features[sig] = self.signal_encoder[sig](batch[sig])
            recon_sig = self.signal_decoder[sig](signal_features[sig])
            if recon_sig.shape != batch[sig].shape: signal_reconstruction[sig] = recon_sig.reshape(batch_size, num_frames, *batch[sig].shape[2:])
            else: signal_reconstruction[sig] = recon_sig
            projected_feat = self.signal_projection[sig](signal_features[sig].reshape(batch_size * num_frames, -1)) # B x T x D
            projected_feature[sig] = rearrange(projected_feat, '(b t) d -> b t d', b=batch_size)

        if 'text' in batch and len(self.signals) >= 7:
            # print('compare shape of different ', batch['text'].shape, projected_feature[self.signals[0]].shape)
            projected_feature['text'] = repeat(batch['text'], 'b d -> b t d', t=num_frames)


        # pooling over signals max over S [projected feature B x T x S x D -> B x T x D]
        activation, hyperplane = self.linear1(projected_feature, batch['context'], x_res=None, p_res=None)
        activation, hyperplane = self.linear2(activation, hyperplane, x_res=projected_feature, p_res=batch['context'])
        activation, hyperplane = self.linear3(activation, hyperplane, x_res=projected_feature, p_res=batch['context'])
        
        activation = self.merge(activation)

        # conditional feature meshing 
        if self.merge_feat == 'concat':
            x_cond = torch.cat([hyperplane, activation], dim=-1)
        elif self.merge_feat == 'add':
            x_cond = hyperplane + activation 
        else:
            x_cond = activation 

        out_dict = {
            'cond_feature': x_cond, 
            'p': hyperplane,
            'a': activation,
            'recon': signal_reconstruction}
        
        return out_dict



class Network(nn.Module):
    def __init__(
        self,
        args, 
        temperature=1.0,
        signal_embedding=256, #TODO make this a dictionary of dimensions to be set up in config
        text_embedding=768,
        condition_type=[],
        cond_drop_chance=0.1,
        aggregation='pool-max', # contact, pool-avg 
        signals=['tactile-glove-left', 'tactile-glove-right', 'myo-emg-left', 'myo-emg-right', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
    ):
        super().__init__()
        self.signals = signals
        self.pool = args.pool
        self.image_size = args.video_model_config['image_size']
        self.cond_drop_chance = cond_drop_chance
        self.aggregation = aggregation
        self.use_single_frame = True if 'ours' in args.video_model else False
        self.use_signal_reconstruction = True
        self.hyper_plane = args.hyperplane

        # self.hyper_plane = 'individual' # individual / global / None

        # self.text_encoder = TextEncoder(model_type=args.text_model)
        
        if self.hyper_plane == 'global': self.signal_encoder = GlobalHyperplane(args, signals=signals)
        elif self.hyper_plane == 'individual': self.signal_encoder = IndividualHyperPlane(args, signals=signals)
        elif self.hyper_plane == 'global-metric-linear': self.signal_encoder = GlobalHyperplaneMetric(args, signals=signals)
        elif self.hyper_plane == 'local-metric-mlp': self.signal_encoder = LocalHyperplaneMetricLayers(args, signals=signals)
        elif self.hyper_plane == 'global-metric-mlp': self.signal_encoder = GlobalHyperplaneMetricLayers(args, signals=signals)
        else:
            raise NotImplementedError
        
        
        if self.aggregation == 'concat':
            self.aggregate = ProjectionMLP(input_dim=projection_config['output_dim'][args.text_model]*len(condition_type), output_dim=projection_config['output_dim'][args.text_model])

        self.video_pipeline = VideoDiffusion(args, use_text=False, model_type=args.video_model)
        
        if self.use_signal_reconstruction: self.recon_loss = nn.MSELoss()
        if self.hyper_plane is not None: self.cos_loss = nn.CosineSimilarity()


        self.recon_loss_weight = 1.0
        self.alignment_loss_weight_a = 0.0
        self.alignment_loss_weight_d = 0.1
        self.diffusion_loss_weight = 10.0

    def _load_pretrained_signal_encoder(self, ckpt_path):
        self.signal_encoder.load_state_dict(torch.load(ckpt_path))

    def encode_action(self, action_data):
        batch_size, num_frames = action_data[self.signals[0]].shape[0], action_data[self.signals[0]].shape[1] - 1
        concat_dim=2 if self.use_single_frame else 1

        loss = 0

        out_dict = self.signal_encoder(action_data)
        # accumulative difference 
        signal_feature = out_dict['cond_feature']

        # print(signal_feature.shape)
        signal_feature = signal_feature * (torch.rand(signal_feature.shape[0], 1, 1, device = signal_feature.device) > self.cond_drop_chance).float()
        # print((torch.rand(signal_feature.shape[0], 1, 1, device = signal_feature.device) > self.cond_drop_chance))

        return signal_feature, out_dict
        
    def sample(self, image, action_data, batch_size=1, other_context={}, negative_task_emb=None):

        signal_feature, out_dict = self.encode_action(action_data)
        image_feat = out_dict['p']

        context_feature = signal_feature

        return self.video_pipeline.sample(image, context_feature, batch_size=batch_size, image_feat=image_feat, negative_task_emb=negative_task_emb)


    def sample_latent(self, image, signal_feature, batch_size=1, other_context={}, negative_task_emb=None):

        context_feature = signal_feature
        return self.video_pipeline.sample(image, context_feature, batch_size=batch_size, image_feat=None, negative_task_emb=negative_task_emb)


    def forward(self, x, signal_data, x_dist, text_feature, negative_task_emb=None):

        batch_size = x.shape[0]
        num_frames = signal_data[self.signals[0]].shape[1]
        # Getting Image and Text Features
        
        signal_features = {}
        projected_feature = {}
        loss_dict = {}
        loss = 0
        recon_loss = 0
        alignment_loss = 0

        pool_feature, out_dict = self.encode_action(signal_data)

        # recon loss
        for sig in self.signals:
            recon_loss += self.recon_loss(out_dict['recon'][sig], signal_data[sig])

        # alignment loss
        video_latent = rearrange(signal_data['video_latent'], '(b t) d -> b t d', b=batch_size)
        video_latent_diff = video_latent[:, 1:, :] - video_latent[:, :1, :]
        # out_dict['a'] = torch.cumsum(out_dict['a'], dim=1)
        alignment_loss_d = self.cos_loss(out_dict['a'], video_latent_diff)

        # magitude loss 
        alignment_loss_a = self.recon_loss(torch.norm(out_dict['a'], dim=-1), torch.mean(rearrange(x_dist, 'b t c h w -> b t (c h w)'), dim=-1)).mean()

        if pool_feature.ndim == 1: pool_feature = pool_feature.unsqueeze(0)

        # pool_feature = pool_feature * (torch.rand(pool_feature.shape[0], 1, 1, device = pool_feature.device) > self.cond_drop_chance).float()
        diffusion_loss, pred = self.video_pipeline(x, pool_feature, image_feat=out_dict['p'],  negative_task_emb=negative_task_emb)
        
        loss = diffusion_loss.mean() * self.diffusion_loss_weight  + \
               recon_loss.mean() * self.recon_loss_weight + \
               alignment_loss_a.mean() * self.alignment_loss_weight_a + \
               alignment_loss_d.mean() * self.alignment_loss_weight_d
              
        
        return loss, pred


