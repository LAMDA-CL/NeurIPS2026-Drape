#!/bin/bash

if [ -z "${GPUS:-}" ]; then
    GPUS="0,1,2,3"
fi
export GPUS

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

# export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
# export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
# export NCCL_NET=${NCCL_NET:-Socket}

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################
python /usr/local/bin/deepspeed --include "localhost:$GPUS" --master_port 29601 llava/train/train_drape.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path data/LLaVa \
    --pretrain_mm_mlp_adapter data/LLaVa/mm_projector.bin \
    --version $PROMPT_VERSION \
    --data_path data/instructions/Instructions_Original/ImageNet/train.json \
    --mm_projector_lr 2e-5 \
    --image_folder data/datasets \
    --vision_tower data/CLIP \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava_drape_CoIN_Task3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --enable_continual_learning True \
    --current_task_name ImageNet \
    --clip_model_name data/CLIP \
    --null_space_threshold 0.99 \
    --train_mm_projector_with_null_space True \
    --previous_task_model_path ./checkpoints/llava_drape_CoIN_Task2
