import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from fairscale.nn.checkpoint import checkpoint_wrapper
from torchvision import models
from exp.video.models.encoders import ImageEncoder
from exp.video.config import image_model_cfg
# from exp.action.models.model_modules import SequentialMLP, ModifiedResNet, ResNet, TactileEncoder, PositionEncoder


from .util import *


USE_TEMPORAL_TRANSFORMER = True


class UNetSD_Our_LCD(nn.Module):
    def __init__(self,
            config=None,
            in_dim=7,
            dim=512,
            y_dim=512,
            context_dim=512,
            hist_dim = 156,
            concat_dim = 8,
            dim_condition=4,
            out_dim=6,
            num_tokens=4,
            frames=8,
            dim_mult=[1, 2, 3, 4],
            num_heads=None,
            head_dim=64,
            num_res_blocks=3,
            attn_scales=[1 / 2, 1 / 4, 1 / 8],
            use_scale_shift_norm=True,
            dropout=0.1,
            temporal_attn_times=1,
            temporal_attention = True,
            use_checkpoint=False,
            use_image_dataset=False,
            use_sim_mask = False,
            use_resnet_conext = False, 
            training=True,
            inpainting=True,
            p_all_zero=0.1,
            p_all_keep=0.1,
            zero_y = None,
            adapter_transformer_layers = 1,
            image_model_choice='resnet',
            **kwargs):
        super(UNetSD_Our_LCD, self).__init__()
        


        embed_dim = dim * 4
        num_heads=num_heads if num_heads else dim//32
        self.horizon = 4
        self.zero_y = zero_y
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.num_tokens = num_tokens
        self.context_dim = context_dim
        self.hist_dim = hist_dim
        self.concat_dim = concat_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        ### for temporal attention
        self.num_heads = num_heads
        ### for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_sim_mask = use_sim_mask
        self.use_resnet_conext = use_resnet_conext
        self.image_model_choice = image_model_choice
        self.training=training
        self.inpainting = inpainting
        self.p_all_zero = p_all_zero
        self.p_all_keep = p_all_keep
        concat_dim = self.in_dim
        self.concat_dim = self.in_dim
        self.frames = frames

        use_linear_in_temporal = False
        transformer_depth = 1
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), # [320,1280]
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        
        ''' 
        self.fps_embedding = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        
        nn.init.zeros_(self.fps_embedding[-1].weight)
        nn.init.zeros_(self.fps_embedding[-1].bias)
        '''
        if temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            self.rotary_emb = RotaryEmbedding(min(32, head_dim))
            self.time_rel_pos_bias = RelativePositionBias(heads = num_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # [Local Image embeding]
        '''
        self.local_image_concat = nn.Sequential(
            nn.Conv2d(3, concat_dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=1, padding=1))
        '''
        self.local_temporal_encoder = TransformerV2(
                heads=2, dim=concat_dim, dim_head_k=concat_dim, dim_head_v=concat_dim, 
                dropout_atte = 0.05, mlp_dim=concat_dim, dropout_ffn = 0.05, depth=adapter_transformer_layers)

        '''
        self.local_image_embedding = nn.Sequential(
            nn.Conv2d(3, concat_dim * 8, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(concat_dim * 8, concat_dim * 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 16, 1024, 3, stride=2, padding=1))
        '''

        # self.image_model_choice = 'resnet-train'
        if self.image_model_choice == 'resnet-train':
            print('train condition model')
            condition_encoder = models.resnet50(pretrained=True)
            condition_encoder = list(condition_encoder.children())
            condition_encoder.pop()
            #self.model_ft =model_ft
            self.condition_encoder = nn.Sequential(*condition_encoder)
        else:
            print('not use condition model')

        # self.image_context_dim = image_model_cfg[image_model_choice]['dimension']
        self.image_context_dim = 1024 if self.image_model_choice != 'resnet-train' else 2048
        
        self.context_embedding = nn.Sequential(
            nn.Linear(self.image_context_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, context_dim // self.horizon))
        

        self.merge_context = nn.Linear(context_dim*2, context_dim)

        # encoder
        self.input_blocks = nn.ModuleList()
        # init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim + concat_dim, dim, 3, padding=1)])
        ####need an initial temporal attention?
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(TemporalTransformer(dim, num_heads, head_dim, depth=transformer_depth, context_dim=context_dim, use_linear=use_linear_in_temporal,  frames=self.frames, multiply_zero=use_image_dataset))
            else:
                init_block.append(TemporalAttentionMultiBlock(dim, num_heads, head_dim, rotary_emb=self.rotary_emb, temporal_attn_times=temporal_attn_times, use_image_dataset=use_image_dataset))
        # elif temporal_conv:
        # init_block.append(InitTemporalConvBlock(dim,dropout=dropout,use_image_dataset=use_image_dataset))
        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(in_dim, embed_dim, dropout, out_channels=out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset,)])
                if scale in attn_scales:
                    # block.append(FlashAttentionBlock(out_dim, context_dim, num_heads, head_dim))
                    block.append(
                            SpatialTransformer(
                                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim, use_linear=True
                            )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(TemporalTransformer(out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb = self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    # block = nn.ModuleList([ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm, 'downsample')])
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim
                    )
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    # block.append(TemporalConvBlock(out_dim,dropout=dropout,use_image_dataset=use_image_dataset))
                    self.input_blocks.append(downsample)
        
        self.middle_block = nn.ModuleList([
            ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False, use_image_dataset=use_image_dataset,),
            SpatialTransformer(
                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim, use_linear=True
            )])        
        
        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                 TemporalTransformer( 
                            out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                            use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset)
                )
            else:
                self.middle_block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =  self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))        

        self.middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        # decoder
        self.output_blocks = nn.ModuleList()
        counter = 0
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):

            for j in range(num_res_blocks + 1):
                # if counter < 11:
                block = nn.ModuleList([ResBlock(in_dim + shortcut_dims.pop(), embed_dim, dropout, out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset, )])
                # else:
                    # block == nn.ModuleList([])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=1024, use_linear=True)
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                                    use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset)
                            )
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)

                self.output_blocks.append(block)
                counter += 1


        
        # assert len(self.input_blocks) == len(self.output_blocks), f'block length: {len(self.input_blocks)} {len(self.output_blocks)}'

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))
        
        # zero out the last layer params
        nn.init.zeros_(self.out[-1].weight)
            

    def forward(self, 
        x,
        t,
        y,
        image = None,
        local_image = None,
        masked = None,
        fps = None,
        video_mask = None,
        focus_present_mask = None,
        prob_focus_present = 0.,  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        mask_last_frame_num = 0,  # mask last frame num
        **kwargs):
        
        assert self.inpainting or masked is None, 'inpainting is not supported'
        assert y is not None

        batch, c, f, h, w= x.shape # x is video 

        # f = f + 1
        device = x.device
        self.batch = batch


        # get first frame 
        if local_image.ndim == 5 and local_image.size(2) > 1:
            local_image = local_image[:, :, :self.horizon, ...]
        elif local_image.ndim != 5:
            local_image = local_image.unsqueeze(2)

        image_context = local_image.clone()

        #### image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)
        else:
            time_rel_pos_bias = None

        # [Concat]
        # concat = x.new_zeros(batch, self.concat_dim, f, h, w)
        # if f > 1:
        #     mask_pos = torch.cat([(torch.ones(local_image[:,:,:1].size())*( (tpos+1)/(f-1) )).cuda() for tpos in range(f-1)], dim=2) # --> f-1
        #     _ximg = torch.cat([local_image[:,:,:1], mask_pos], dim=2)
        #     _ximg = rearrange(_ximg, 'b c f h w -> (b f) c h w')
        # else:
            # _ximg = rearrange(local_image, 'b c f h w -> (b f) c h w')
        _ximg = rearrange(local_image.repeat(1, 1, f//self.horizon, 1, 1), 'b c f h w -> (b f) c h w')
        # _ximg # b x f, c, w, h # leading local image with noise scale for the rest of the frames
            
        # _ximg = self.local_image_concat(_ximg) # b x f, c, w, h

        _h = _ximg.shape[2]
        _ximg = rearrange(_ximg, '(b f) c h w -> (b h w) f c', b = batch)
        _ximg = self.local_temporal_encoder(_ximg) # b x w x h, f, c
        _ximg = rearrange(_ximg, '(b h w) f c -> b c f h w', b = batch, h = _h)
        concat = _ximg   # b, c, f, h, w 
        
        # [Embeddings]
        embeddings = self.time_embed(sinusoidal_embedding(t, self.dim)) 
        # TODO: add fps embeeding
        #+ self.fps_embedding(sinusoidal_embedding(fps, self.dim))
        embeddings = embeddings.repeat_interleave(repeats=f, dim=0)

        # [Context]
        # [C] for text input
        context = x.new_zeros(batch, 0, self.context_dim)

        # b, 0, dim
        y_context = y
        context = torch.cat([context, y_context], dim=1) # b t self.context_dim
        

        # DIFF: ADD IMAGE CONTEXT
        # image_context = self.condition_encoder(rearrange(image_context, 'b c t h w -> (b t) c h w')).squeeze() # b x c
        if self.image_model_choice == 'resnet-train': image_context = self.condition_encoder(rearrange(image_context, 'b c t h w -> (b t) c h w')).squeeze() 
        elif image is not None: image_context = image
        else: raise NotImplementedError # b x c
        
        context_emb = self.context_embedding(image_context)
        if self.horizon > 1: context_emb = rearrange(context_emb, '(b h) d -> b (h d)', h=self.horizon)
        
        if image_context.ndim == 1: image_context = image_context.unsqueeze(0)
        context_emb = repeat(context_emb, 'b d -> b f d', f=context.shape[1])

        ## TODO add attention here between local image context and global context
        
        context = torch.cat([context, context_emb], dim=-1) 

        context = self.merge_context(rearrange(context, 'b f d -> (b f) d'))

        context = context.unsqueeze(1)

        # context = rearrange(context, 'b f d -> (b f) 1 d')

        # context : b, b, dim

        # [C] for local input
        # TODO: # remove local image context
        # local_context = rearrange(local_image, 'b c f h w -> (b f) c h w')
        # local_context = self.local_image_embedding(local_context) # b, dim, 8, 8

        # print('local context', local_context.shape)
        
        # h = local_context.shape[2]
        # local_context = rearrange(local_context, 'b c h w -> b (h w) c', b = batch, h = h) # [b, 64, dim]
        # context = torch.cat([context, local_context], dim=1) # [b, 64 + b, dim]

        # print('context + localcontext', context.shape)
    

        # [C] for global input

        # context = context.repeat_interleave(repeats=f, dim=0) # b x f, 64 + b, dim
        # context = rearrange(context, '(b f) c d -> b f c d', f=f)

        
        x = torch.cat([x, concat], dim=1)
    
        x = rearrange(x, 'b c f h w -> (b f) c h w')


        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, embeddings, context, time_rel_pos_bias, focus_present_mask, video_mask)
            xs.append(x)

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, embeddings, context, time_rel_pos_bias, focus_present_mask, video_mask)
        
        # decoder
        for b_ind, block in enumerate(self.output_blocks):
            xs_cond = xs.pop()
            ref = xs[-1] if len(xs) > 0 else xs_cond
            x = torch.cat([x, xs_cond], dim=1)
            # print(b_ind, ref.shape)
            x = self._forward_single(block, x, embeddings, context, time_rel_pos_bias, focus_present_mask, video_mask, reference=ref)
        
        # head
        x = self.out(x) # [32, 4, 32, 32]
        
        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b = batch)
        return x
        

    
    def _forward_single(self, module, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference=None):
        if isinstance(module, ResidualBlock):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            # xs = x.clone()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            xn = module(x, context)
            x = rearrange(xn, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalTransformer_attemask):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, MemoryEfficientCrossAttention):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, Upsample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x)
        elif isinstance(module, Downsample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x)
        elif isinstance(module, Resample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, reference)
        elif isinstance(module, TemporalAttentionBlock):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalAttentionMultiBlock):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, InitTemporalConvBlock):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalConvBlock):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block,  x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference)
        else:
            x = module(x)
        return x