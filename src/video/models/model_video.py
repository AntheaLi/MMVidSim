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

from exp.video.models.encoders import SignalEncoder, TextEncoder, SignalFrameEncoder, SignalDecoder, SignalFrameDecoder
from exp.action.models.model_modules import *
from exp.video.action_config import projection_config, loss_weight_config

from .modules.i2vgen.autoencoder import AutoencoderKL
from .modules.i2vgen.unet import UNetSD_I2VGen
from .modules.i2vgen.unet.unet_ours import UNetSD_Our
from .modules.i2vgen.unet.unet_ours_lcd import UNetSD_Our_LCD
from .modules.i2vgen.diffusion.diffusion_ddim import DiffusionDDIM
from .modules.flowdiffusion.goal_diffusion import GoalGaussianDiffusion
from .modules.flowdiffusion.goal_diffusion_i2v import GoalGaussianDiffusionI2V
from .modules.flowdiffusion.goal_diffusion_ours_lcd import GoalGaussianDiffusionLCD
from .modules.flowdiffusion.unet import Unet64
from .modules.sd.diffusionmodules.video_model import VideoUNet
from config import video_model_cfg

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
            self.unet = UNetSD_Our_LCD(**args.video_model_config['our']['unet'])
            self.pipeline = GoalGaussianDiffusionLCD(self.unet, **args.video_model_config['i2vgen-avdc'])    
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

    def sample(self, image, action_data, batch_size=1, negative_task_emb=None):
        if self.model_type == 'i2vgen':
            image = image[:,:, 0:1].detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, negative_task_emb=negative_task_emb, guide_scale=self.guide_scale)
        elif self.model_type == 'i2vgen-avdc':
            image = image[:,:, 0:1].detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, guide_scale=self.guide_scale)
        elif self.model_type == 'ours-lcd':
            if self.use_text: action_data = repeat(action_data, 'b c -> b f c', f=self.num_frames)
            image = image[:,:, 0:1].detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, guide_scale=self.guide_scale)
        else:
            image = image.detach().clone()
        return self.pipeline.sample(image, action_data, batch_size=batch_size, guide_scale=self.guide_scale)

        
    def forward(self, x, txt_embedding, negative_task_emb=None):
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
            
            loss, model_pred = self.pipeline(target, image, txt_embedding, negative_task_emb=negative_task_emb)

        
        elif self.model_type == 'i2vgen-avdc':
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            loss, model_pred = self.pipeline(target, image, txt_embedding)

        elif self.model_type == 'ours': #TODO
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            loss, model_pred = self.pipeline(target, image, txt_embedding)

        elif self.model_type == 'ours-lcd': #TODO
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            if self.use_text: txt_embedding = repeat(txt_embedding, 'b c -> b f c', f=t-1)

            loss, model_pred = self.pipeline(target, image, txt_embedding)


        elif self.model_type == 'sd-avdc':
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            loss, model_pred = self.pipeline(target, image, txt_embedding)

        elif self.model_type == 'i2vgen-xl':
            generator = torch.manual_seed(8888)
            model_pred = self.pipeline(prompt=x['text_label'],image=image,generator=generator).frames[0]
            loss = F.mse_loss(model_pred, target, reduction="mean")
            
        elif self.model_type in ['avdc', 'flowdiffusion']:
            
            target = target.reshape(b, (t-1) * c, h, w)
            loss, model_pred = self.pipeline( target, image, txt_embedding)
        
        else:       
            loss, model_pred = self.pipeline( target, image, txt_embedding)

        return loss, model_pred



class LatentVideoDiffusion(nn.Module):
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
            self.ae = AutoencoderKL(**args.video_model_config['i2vgen']['ae'])
            self.unet = UNetSD_I2VGen(**args.video_model_config['i2vgen']['unet'])
            self.pipeline = DiffusionDDIM(self.unet, **args.video_model_config['i2vgen']['diffusion'])    
        elif self.model_type == 'i2vgen-avdc':
            self.ae = AutoencoderKL(**args.video_model_config['i2vgen']['ae'])
            self.unet = UNetSD_I2VGen(**args.video_model_config['i2vgen']['unet'])
            self.pipeline = GoalGaussianDiffusionI2V(self.unet, **args.video_model_config['i2vgen-avdc'])    
        elif self.model_type == 'ours':
            self.ae = AutoencoderKL(**args.video_model_config['i2vgen']['ae'])
            self.unet = UNetSD_Our(**args.video_model_config['our']['unet'])
            self.pipeline = GoalGaussianDiffusionI2V(self.unet, **args.video_model_config['i2vgen-avdc'])    
        elif self.model_type in ['avdc', 'flowdiffusion']:
            args.video_model_config['avdc']['model'] = Unet64()
            self.pipeline = GoalGaussianDiffusion(**args.video_model_config['avdc'])    
        else:
            args.video_model_config['avdc']['model'] = Unet64()
            self.pipeline = GoalGaussianDiffusion(**args.video_model_config['avdc'])

    def sample(self, image, action_data, batch_size=1, negative_task_emb=None):
        batch_size = image.shape[0]
        if self.model_type == 'i2vgen':
            image = self.ae.encode(rearrange(image[:,:, 0:1], 'b c t h w -> (b t) c h w'))
            image = rearrange(image, '(b t) c h w -> b c t h w', b=batch_size).detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, negative_task_emb=negative_task_emb, guide_scale=self.guide_scale)
        elif self.model_type == 'i2vgen-avdc':
            image = self.ae.encode(rearrange(image[:,:, 0:1], 'b c t h w -> (b t) c h w'))
            image = rearrange(image, '(b t) c h w -> b c t h w', b=batch_size).detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, guide_scale=self.guide_scale)
        elif self.model_type == 'our':
            image = self.ae.encode(rearrange(image[:,:, 0:1], 'b c t h w -> (b t) c h w'))
            image = rearrange(image, '(b t) c h w -> b c t h w', b=batch_size).detach().clone()
            return self.pipeline.sample(image, action_data, batch_size=batch_size, guide_scale=self.guide_scale)
        else:
            image = image.detach().clone()
        return self.pipeline.sample(image, action_data, batch_size=batch_size, guide_scale=self.guide_scale)

        
    def forward(self, x, txt_embedding, negative_task_emb=None):
        x = x.permute(0, 1, 4, 2, 3)
        b, t, c, h, w = x.shape
        image = x[:,0:1].squeeze().detach().clone()
        target = x[:, 1:]
    
     
        if self.model_type == 'i2vgen':
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]

            image = self.ae.encode(rearrange(image, 'b c t h w -> (b t) c h w'))
            image = rearrange(image, '(b t) c h w -> b c t h w', b=b)
            target = self.ae.encode(rearrange(target, 'b c t h w -> (b t) c h w'))
            target = rearrange(target, '(b t) c h w -> b c t h w', b=b)

            loss, model_pred = self.pipeline(target, image, txt_embedding, negative_task_emb=negative_task_emb)

            model_pred = self.ae.decode(rearrange(model_pred, 'b c t h w -> (b t) c h w'))

            model_pred = rearrange(model_pred, '(b t) c h w -> b c t h w', b=b)

        elif self.model_type == 'i2vgen-avdc':
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            image = self.ae.encode(rearrange(image, 'b c t h w -> (b t) c h w'))
            image = rearrange(image, '(b t) c h w -> b c t h w', b=b)
            target = self.ae.encode(rearrange(target, 'b c t h w -> (b t) c h w'))
            target = rearrange(target, '(b t) c h w -> b c t h w', b=b)

            loss, model_pred = self.pipeline(target, image, txt_embedding)

            model_pred = self.ae.decode(rearrange(model_pred, 'b c t h w -> (b t) c h w'))

            model_pred = rearrange(model_pred, '(b t) c h w -> b c t h w', b=b)

        elif self.model_type == 'ours': #TODO
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            image = self.ae.encode(rearrange(image, 'b c t h w -> (b t) c h w'))
            image = rearrange(image, '(b t) c h w -> b c t h w', b=b)
            target = self.ae.encode(rearrange(target, 'b c t h w -> (b t) c h w'))
            target = rearrange(target, '(b t) c h w -> b c t h w', b=b)

            loss, model_pred = self.pipeline(target, image, txt_embedding)

            model_pred = self.ae.decode(rearrange(model_pred, 'b c t h w -> (b t) c h w'))

            model_pred = rearrange(model_pred, '(b t) c h w -> b c t h w', b=b)

        elif self.model_type == 'ours-lcd': #TODO
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            image = self.ae.encode(rearrange(image, 'b c t h w -> (b t) c h w'))
            image = rearrange(image, '(b t) c h w -> b c t h w', b=b)
            target = self.ae.encode(rearrange(target, 'b c t h w -> (b t) c h w'))
            target = rearrange(target, '(b t) c h w -> b c t h w', b=b)

            loss, model_pred = self.pipeline(target, image, txt_embedding)

            model_pred = self.ae.decode(rearrange(model_pred, 'b c t h w -> (b t) c h w'))

            model_pred = rearrange(model_pred, '(b t) c h w -> b c t h w', b=b)

        elif self.model_type == 'sd-avdc':
            if image.ndim == 4: image = image.unsqueeze(2)
            elif image.ndim == 5: image = image.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            assert target.shape[1] == image.shape[1]
            
            loss, model_pred = self.pipeline(target, image, txt_embedding)

        elif self.model_type == 'i2vgen-xl':
            generator = torch.manual_seed(8888)
            model_pred = self.pipeline(prompt=x['text_label'],image=image,generator=generator).frames[0]
            loss = F.mse_loss(model_pred, target, reduction="mean")
            
        elif self.model_type in ['avdc', 'flowdiffusion']:
            
            target = target.reshape(b, (t-1) * c, h, w)
            loss, model_pred = self.pipeline( target, image, txt_embedding)
        
        else:       
            loss, model_pred = self.pipeline( target, image, txt_embedding)

        return loss, model_pred



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
        signals=['tactile-glove-left', 'tactile-glove-right', 'myo-left', 'myo-right', 'joint-rotation', 'joint-position', 'right-hand-pose', 'left-hand-pose'],
    ):
        super().__init__()
        self.signals = signals
        self.pool = args.pool
        self.image_size = args.video_model_config['image_size']
        self.cond_drop_chance = cond_drop_chance
        self.aggregation = aggregation
        self.use_single_frame = True if 'ours' in args.video_model else False
        self.use_signal_reconstruction = True

        # self.text_encoder = TextEncoder(model_type=args.text_model)

        self.signal_encoder = nn.ModuleDict()
        self.signal_projection = nn.ModuleDict()
        self.signal_decoder = nn.ModuleDict()


        for sig in signals:
            if 'our' in args.video_model: 
                self.signal_encoder[sig] = SignalFrameEncoder(args, sig)
                if self.use_signal_reconstruction: self.signal_decoder[sig] = SignalFrameDecoder(args, sig)
            else: 
                self.signal_encoder[sig] = SignalEncoder(args, sig)
                if self.use_signal_reconstruction: self.signal_decoder[sig] = SignalDecoder(args, sig)
                
            self.signal_projection[sig] = ProjectionMLP(input_dim=projection_config[sig][args.signal_model_config[sig]], output_dim=projection_config['output_dim'][args.text_model])
            # self.signal_decoder[sig] = SignalDecoder(args, sig)

        if self.aggregation == 'concat':
            self.aggregate = ProjectionMLP(input_dim=projection_config['output_dim'][args.text_model]*len(condition_type), output_dim=projection_config['output_dim'][args.text_model])

        self.video_pipeline = VideoDiffusion(args, use_text=False, model_type=args.video_model)
        
        if self.use_signal_reconstruction: self.recon_loss = nn.MSELoss(reduction='mean')

    def encode_action(self, action_data):
        batch_size, num_frames = action_data[self.signals[0]].shape[0], action_data[self.signals[0]].shape[1] - 1
        concat_dim=2 if self.use_single_frame else 1

        signal_features = {}
        projected_feature = {}
        signal_decode = {}
        recon_loss = 0

        for sig in self.signals:
            action_dim, action_data_dim = action_data[sig].shape[2:], action_data[sig][:, 1:].shape
            signal_features[sig] = self.signal_encoder[sig](action_data[sig][:, 1:])
            if self.use_signal_reconstruction: 
                signal_decode[sig] = self.signal_decoder[sig](signal_features[sig]).reshape(*action_data_dim)
                recon_loss += self.recon_loss(action_data[sig][:,1:], signal_decode[sig]).mean()
            if self.use_single_frame: signal_features[sig] = signal_features[sig].reshape(batch_size * num_frames, -1)
            else: signal_features[sig] = signal_features[sig].reshape(batch_size, -1)

            projected_feature[sig] = self.signal_projection[sig](signal_features[sig])
            if self.use_single_frame: projected_feature[sig] = projected_feature[sig].reshape(batch_size, num_frames, -1)


        if self.aggregation == 'pool-avg': signal_feature = torch.stack([projected_feature[x].unsqueeze(concat_dim) for x in self.signals], dim=concat_dim).mean(concat_dim).squeeze(concat_dim)
        elif self.aggregation == 'pool-max': signal_feature = torch.stack([projected_feature[x].unsqueeze(concat_dim) for x in self.signals], dim=concat_dim).max(concat_dim)[0].squeeze(concat_dim)
        elif self.aggregation == 'add': signal_feature = torch.stack([projected_feature[x].unsqueeze(concat_dim) for x in self.signals], dim=concat_dim).sum(dim=concat_dim).squeeze(concat_dim)
        else: signal_feature = torch.stack([projected_feature[x].unsqueeze(concat_dim) for x in self.signals], dim=concat_dim).max(concat_dim)[0].squeeze(concat_dim) # default pool

        if signal_feature.ndim == 1: signal_feature = signal_feature.unsqueeze(0)
        if self.use_single_frame and signal_feature.ndim == 2: signal_feature = signal_feature.unsqueeze(0)

        # print(signal_feature.shape)
        signal_feature = signal_feature * (torch.rand(signal_feature.shape[0], 1, 1, device = signal_feature.device) > self.cond_drop_chance).float()
        # print((torch.rand(signal_feature.shape[0], 1, 1, device = signal_feature.device) > self.cond_drop_chance))

        return signal_feature, recon_loss
    
    def aggregate_feature(self, signal_feature, other_context={}):
        if other_context == {}:
            return signal_feature
        
        if self.aggregation == 'concat':
            context_feature = torch.cat([signal_feature, *[other_context[key] for key in other_context]], dim=-1)
            context_feature = self.aggregate(context_feature)
        elif self.aggregation == 'pool-avg': context_feature = torch.stack([signal_feature, *[other_context[key] for key in other_context]], dim=1).mean(1).squeeze()
        elif self.aggregation == 'pool-max': context_feature = torch.stack([signal_feature, *[other_context[key] for key in other_context]], dim=1).max(1)[0].squeeze()
        else: context_feature = torch.stack([signal_feature, *[other_context[key] for key in other_context]], dim=1).max(1)[0].squeeze()
        return context_feature

    def sample(self, image, action_data, batch_size=1, other_context={}, negative_task_emb=None):

        signal_feature, _ = self.encode_action(action_data)

        print(signal_feature.shape)

        context_feature = signal_feature

        return self.video_pipeline.sample(image, context_feature, batch_size=batch_size, negative_task_emb=negative_task_emb)

    def forward(self, x, signal_data, negative_task_emb=None):

        batch_size = x.shape[0]
        # Getting Image and Text Features
        
        signal_features = {}
        projected_feature = {}
        loss_dict = {}
        loss = 0

        pool_feature, recon_loss = self.encode_action(signal_data)
        

        if pool_feature.ndim == 1: pool_feature = pool_feature.unsqueeze(0)

        # pool_feature = pool_feature * (torch.rand(pool_feature.shape[0], 1, 1, device = pool_feature.device) > self.cond_drop_chance).float()

        loss, pred = self.video_pipeline(x, pool_feature, negative_task_emb=negative_task_emb)
        
        loss = loss + recon_loss
        
        return loss, pred


