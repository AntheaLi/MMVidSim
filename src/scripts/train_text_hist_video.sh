gpu=6
text_model="open_clip"
diffusion_model="ours-lcd"
lcdtrain="resnet-train"
image_size=64
guide_scale=0.0

CUDA_VISIBLE_DEVICES=${gpu} python train_text_hist_video.py \
    --exp-suffix exp-text-hist-${diffusion_model}-${image_size} \
    --guide-scale ${guide_scale} \
    --video-model ${diffusion_model} \
    --text-model ${text_model} \
    --lcd-ctx-model ${lcdtrain} \
    --log-dir "log-hist" \
    --batch-size 16 \
    --eval-batch-size 5

