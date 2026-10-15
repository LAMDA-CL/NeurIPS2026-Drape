#!/bin/bash

gpu_list="0,1,2,3"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    gpu_list="$CUDA_VISIBLE_DEVICES"
fi
if [ -n "$3" ] ;then
    gpu_list="$3"
fi
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='task6'
else
    STAGE=$1
fi

MODELPATH="${2:-checkpoints/llava_drape_UCIT_Task6}"

RESULT_DIR="./results/UCIT/VizWiz"
mkdir -p "$RESULT_DIR/$STAGE"

for IDX in $(seq 0 $((CHUNKS-1))); do
CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python -m llava.eval.Eval.model_answer \
    --model-path $MODELPATH \
    --model-base data/LLaVa \
    --question-file data/instructions/UCIT/VizWiz/test_3000.json \
    --image-folder data/dataset_UCIT \
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


python -m llava.eval.Eval.eval_caption \
    --annotation-file data/instructions/UCIT/VizWiz/val_coco_type_3000.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \
