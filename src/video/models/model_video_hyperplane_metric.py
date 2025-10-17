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
from .modules.flowdiffusion.goal_diffusion_ours_lcd import GoalGaussianDiffusionLCD
from .modules.flowdiffusion.goal_diffusion_i2v import GoalGaussianDiffusionI2V
from .modules.flowdiffusion.unet import Unet64
from .modules.sd.diffusionmodules.video_model import VideoUNet
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

        for sig in signals:
            self.signal_encoder[sig] = SignalFrameEncoder(args, sig)
            self.signal_projection[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_decoder[sig] = SignalFrameDecoder(args, sig)


        self.signal_direction = nn.Linear(projection_config['output_dim'][args.text_model], projection_config['output_dim'][args.text_model], bias=False)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.negative_slope = 0.0
        self.merge_feat = 'activation'

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
        action_feature = torch.stack([projected_feature[x].unsqueeze(expand_dim) for x in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D

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

        self.signal_encoder = nn.ModuleDict()
        self.signal_projection = nn.ModuleDict()
        self.signal_direction = nn.ModuleDict()
        self.signal_decoder = nn.ModuleDict()

        for sig in signals:
            self.signal_encoder[sig] = SignalFrameEncoder(args, sig)
            self.signal_projection[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_direction[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_decoder[sig] = SignalFrameDecoder(args, sig)


        self.mse_loss = nn.MSELoss(reduction='mean')
        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.negative_slope = 0.0
        self.merge_feat = 'add'


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
        signal_activation_max = torch.stack([signal_activation[x].unsqueeze(expand_dim) for x in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D        


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
class NetworkGlobalHyperplaneMetric(nn.Module):
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
        self.signal_decoder = nn.ModuleDict()
        

        for sig in signals:
            self.signal_encoder[sig] = SignalFrameEncoder(args, sig)
            self.signal_projection[sig] = nn.Linear(projection_config[sig][args.signal_model_config[sig]], projection_config['output_dim'][args.text_model], bias=False)
            self.signal_decoder[sig] = SignalFrameDecoder(args, sig)

        self.signal_direction = nn.Sequential(nn.Linear(2048, projection_config['output_dim'][args.text_model]), 
                                            nn.LeakyReLU(), 
                                            nn.Linear(projection_config['output_dim'][args.text_model], projection_config['output_dim'][args.text_model], bias=False))
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.negative_slope = 0.0
        self.merge_feat = 'add'
        self.cos_loss_dir = 1.0
        self.cos_loss_magnitude = 1.0

    def forward(self, batch, other_context=None):

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


        # pooling over signals max over S [projected feature B x T x S x D -> B x T x D]
        expand_dim = 1
        action_feature = torch.stack([projected_feature[x].unsqueeze(expand_dim) for x in self.signals], dim=expand_dim).max(expand_dim)[0].squeeze() # B x T x D

        # get plane normal direction
        assert other_context is not None
        p = self.signal_direction(other_context)

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
        
        if self.hyper_plane == 'global': self.signal_encoder = GlobalHyperplane(args, signals)
        elif self.hyper_plane == 'global-metric': self.signal_encoder = GlobalHyperplane(args, signals)
        elif self.hyper_plane == 'individual': self.signal_encoder = IndividualHyperPlane(args, signals)
        else: pass


        if self.aggregation == 'concat':
            self.aggregate = ProjectionMLP(input_dim=projection_config['output_dim'][args.text_model]*len(condition_type), output_dim=projection_config['output_dim'][args.text_model])

        self.video_pipeline = VideoDiffusion(args, use_text=False, model_type=args.video_model)
        
        if self.use_signal_reconstruction: self.recon_loss = nn.MSELoss()
        if self.hyper_plane is not None: self.cos_loss = nn.CosineSimilarity()


        self.recon_loss_weight = 1.0
        self.alignment_loss_weight = 1.0
        self.diffusion_loss_weight = 10.0
        self.metric = False
        
        if self.hyper_plane == 'global-metric': self.alignment_loss_weight = 1e-2

    def _load_pretrained_signal_encoder(self, ckpt_path):
        self.signal_encoder.load_state_dict(torch.load(ckpt_path))

    def encode_action(self, action_data, other_context={}):
        batch_size, num_frames = action_data[self.signals[0]].shape[0], action_data[self.signals[0]].shape[1] - 1
        concat_dim=2 if self.use_single_frame else 1

        loss = 0

        if self.hyper_plane == 'global-metric': out_dict = self.signal_encoder(action_data, other_context=other_context)
        else: out_dict = self.signal_encoder(action_data)
        signal_feature = out_dict['cond_feature']

        # print(signal_feature.shape)
        signal_feature = signal_feature * (torch.rand(signal_feature.shape[0], 1, 1, device = signal_feature.device) > self.cond_drop_chance).float()
        # print((torch.rand(signal_feature.shape[0], 1, 1, device = signal_feature.device) > self.cond_drop_chance))

        return signal_feature, out_dict
        
    def sample(self, image, action_data, batch_size=1, other_context={}, negative_task_emb=None):

        signal_feature, _ = self.encode_action(action_data, other_context=other_context)

        context_feature = signal_feature

        return self.video_pipeline.sample(image, context_feature, batch_size=batch_size, negative_task_emb=negative_task_emb)

    def forward(self, x, signal_data, x_dist, text_feature, other_context=None, negative_task_emb=None):

        batch_size = x.shape[0]
        num_frames = signal_data[self.signals[0]].shape[1]
        # Getting Image and Text Features
        
        signal_features = {}
        projected_feature = {}s
        loss_dict = {}
        loss = 0
        recon_loss = 0
        alignment_loss = 0

        pool_feature, out_dict = self.encode_action(signal_data, other_context=other_context)

        # recon loss
        for sig in self.signals:
            recon_loss += self.recon_loss(out_dict['recon'][sig], signal_data[sig])

        # alignment loss
        if self.hyper_plane == 'global-metric':
            alignment_loss = self.recon_loss(torch.norm(out_dict['a'], dim=-1), torch.mean(rearrange(x_dist, 'b t c h w -> b t (c h w)'), dim=-1)).mean()
        else:
            alignment_loss = self.cos_loss(out_dict['p'], text_feature)
            if self.metric: 
                alignment_loss_a = self.recon_loss(torch.norm(out_dict['a'], dim=-1), torch.mean(rearrange(x_dist, 'b t c h w -> b t (c h w)'), dim=-1)).mean()
                loss += alignment_loss_a.mean() * 1e-2
            

        # magitude loss 

        if pool_feature.ndim == 1: pool_feature = pool_feature.unsqueeze(0)

        # pool_feature = pool_feature * (torch.rand(pool_feature.shape[0], 1, 1, device = pool_feature.device) > self.cond_drop_chance).float()

        diffusion_loss, pred = self.video_pipeline(x, pool_feature, negative_task_emb=negative_task_emb)
        
        loss = diffusion_loss.mean() * self.diffusion_loss_weight  + \
              recon_loss.mean() * self.recon_loss_weight + \
              alignment_loss.mean() * self.alignment_loss_weight + \
        
        return loss, pred


