#!/bin/bash

gpu_list="0,1,2,3"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    gpu_list="$CUDA_VISIBLE_DEVICES"
fi
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='MoELoRA'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='checkpoints/llava_drape_CoIN_Task8'
else
    MODELPATH=$2
fi


RESULT_DIR="./results/GQA"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python -m llava.eval.Eval.model_others \
        --model-path $MODELPATH \
        --model-base data/LLaVa \
        --question-file data/instructions/Instructions_Original/GQA/test.json \
        --image-folder data/datasets \
                --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1  \
    --use-router &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python ./scripts/convert_gqa_for_eval.py \
    --src $output_file \
    --dst $RESULT_DIR/$STAGE/testdev_balanced_predictions.json

python -m llava.eval.Eval.eval_gqa \
    --tier testdev_balanced \
    --path $RESULT_DIR/$STAGE \
    --question-dir data/instructions/Instructions_Original/GQA/ \
    --output-dir $RESULT_DIR/$STAGE
