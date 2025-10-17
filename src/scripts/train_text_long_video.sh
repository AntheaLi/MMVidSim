gpu=4
text_model="open_clip"
diffusion_model="ours-lcd"
lcdtrain="resnet-train"
image_size=64
guide_scale=0.0

CUDA_VISIBLE_DEVICES=${gpu} python train_text_long_video.py \
    --exp-suffix exp-long-text-${diffusion_model}-${image_size} \
    --guide-scale ${guide_scale} \
    --video-model ${diffusion_model} \
    --text-model ${text_model} \
    --lcd-ctx-model ${lcdtrain} \
    --batch-size 18 \
    --eval-batch-size 5 \
    --ckpt "/data/vision/torralba/scratch/yichenl/projects/source/act/exp/video/logs/exp-long-text-ours-lcd-64-2024-05-17-00:34:36.172497/model-4.pt"

