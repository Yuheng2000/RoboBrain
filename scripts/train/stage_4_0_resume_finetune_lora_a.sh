############### Finetune ################
export PYTHONPATH=$(pwd)
export BASE_RUN_NAME=resume_finetune_Llava-Onevision-siglip-qwen2.5-stage-4_lora_affordance

export PREV_STAGE_CHECKPOINT=/home/vlm/workspace/checkpoints/resume_finetune_Llava-Onevision-siglip-qwen2.5-stage-3_robo
export VISION_MODEL_VERSION=/home/vlm/pretrain_model/siglip-so400m-patch14-384

export DATA_PATH=/path/to/stage_4_affordance.yaml

export IMAGE_FOLDER=/home/vlm/train_images
export VIDEO_FOLDER=/home/vlm/train_videos
export OUTPUT_DIR=/home/vlm/workspace/checkpoints/${BASE_RUN_NAME}

export PROMPT_VERSION=qwen_2

export IMAGE_ASPECT_RATIO=anyres_max_9
export MM_TUNABLE_PARTS="lora"
export IMAGE_GRID_PINPOINTS="(1x1),...,(6x6)"

export NUM_GPUS=8
export NNODES=16
export HOSTFILE=/home/vlm/workspace/hostfile/hostfile_group16

export DATA_WORKERS=4
export DEV_BATCHSIZE=1
export GRAD_ACC_STEPS=2

export LEARNING_RATE=1e-5
export VIT_LEARNING_RATE=2e-6
export MAX_SEQ_LEN=32768
export MAX_FRAME_NUM=32
export ZERO_VERSION=3

export WANDB_MODE=offline

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

deepspeed --num_gpus $NUM_GPUS --num_nodes $NNODES --hostfile $HOSTFILE \
    train/train_mem.py \
    --deepspeed scripts/zero${ZERO_VERSION}.json \
    --lora_enable True \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts $MM_TUNABLE_PARTS \
    --mm_vision_tower_lr $VIT_LEARNING_RATE \
    --vision_tower $VISION_MODEL_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio $IMAGE_ASPECT_RATIO \
    --image_grid_pinpoints "$IMAGE_GRID_PINPOINTS" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $DEV_BATCHSIZE \
    --per_device_eval_batch_size $DEV_BATCHSIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MAX_SEQ_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATA_WORKERS \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound $MAX_FRAME_NUM  \
    2>&1 | tee $OUTPUT_DIR/train.log

# You can delete the sdpa attn_implementation if you want to use flash attn
