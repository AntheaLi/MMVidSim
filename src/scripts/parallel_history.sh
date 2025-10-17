export ngpu=${1:-3}

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


torchrun --nproc_per_node=${ngpu} train_history_parallel.py \
    --exp-suffix acc-double-attn-${image_model}-${lcdtrain}-parsep-hyperplane-multi-activation-open-clip-${image_size}-${merge_func}-${hyperplane}-${diffusion_model}-${pool}-${tactile_model}-${position_encoder}-1 \
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
    --eval-freq 0.5 \
    --batch-size 2 \
    --eval-batch-size 2 \





