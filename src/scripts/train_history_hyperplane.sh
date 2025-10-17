gpu=0
position_encoder="mlp-frame"
text_model="open_clip"
image_model="open_clip"
tactile_model="resnet"
diffusion_model="ours-lcd"
hyperplane="global-metric-mlp"
lcdtrain="resnet-train"
image_size=128
guide_scale=0.0
merge_func="softmax"

CUDA_VISIBLE_DEVICES=${gpu} python train_history_hyperplane_video.py \
    --exp-suffix double-attn-${image_model}-${lcdtrain}-parsep-hyperplane-multi-activation-open-clip-${image_size}-${merge_func}-${hyperplane}-${diffusion_model}-${pool}-${tactile_model}-${position_encoder}-1 \
    --data-path "Dataset" \
    --lcd-ctx-model ${lcdtrain} \
    --guide-scale ${guide_scale} \
    --video-model ${diffusion_model} \
    --text-model ${text_model} \
    --tactile-model ${tactile_model} \
    --image-model ${image_model} \
    --joint-position-model ${position_encoder} \
    --hand-pose-model ${position_encoder} \
    --joint-rotation-model ${position_encoder} \
    --myo ${position_encoder} \
    --hyperplane ${hyperplane} \
    --merge-local ${merge_func} \
    --log-dir "log-hist" \
    --eval-freq 1 \
    --batch-size 1 \
    --eval-batch-size 6 \


#    --ckpt "/data/vision/torralba/scratch/yichenl/projects/source/act/exp/video/log-hist/body-joint-model-last.pt" 


#    --ckpt "/data/vision/torralba/scratch/yichenl/projects/source/act/exp/video/logs/exp-no-text-open_clip-resnet-train-parsep-hyperplane-multi-activation-open-clip-64-max-global-metric-mlp-ours-lcd--resnet-mlp-frame-1-2024-05-14-22:11:18.843453/model-9.pt"



#    --ckpt "/data/vision/torralba/scratch/yichenl/projects/source/act/exp/video/logs/exp-open_clip-resnet-train-parsep-hyperplane-multi-activation-open-clip-64-softmax-global-metric-mlp-ours-lcd--resnet-mlp-frame-1-2024-05-10-03:01:37.462308/model-5.pt" \
#    --eval-batch-size 6


#    --pretrain ./logs/pretrain_signal_net/${hyperplane}_alignment.pth \

