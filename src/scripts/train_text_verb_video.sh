gpu=3
text_model="open_clip"
diffusion_model="ours-lcd"
lcdtrain="resnet-train"
image_size=64
guide_scale=0.0

CUDA_VISIBLE_DEVICES=${gpu} python train_text_verb_video.py \
    --exp-suffix exp-verb-text-${diffusion_model}-${image_size} \
    --guide-scale ${guide_scale} \
    --video-model ${diffusion_model} \
    --text-model ${text_model} \
    --lcd-ctx-model ${lcdtrain} \
    --batch-size 18 \
    --eval-batch-size 5 \


#    --ckpt "/data/vision/torralba/scratch/yichenl/projects/source/act/exp/video/logs/exp-verb-text-ours-lcd-64-2024-05-17-00:54:55.056801/model-4.pt"

