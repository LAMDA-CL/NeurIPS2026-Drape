#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH

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


RESULT_DIR="./results/ScienceQA"
mkdir -p "$RESULT_DIR/$STAGE"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python -m llava.eval.Eval.model_science_qa \
        --model-path $MODELPATH \
        --model-base data/LLaVa \
        --question-file data/instructions/Instructions_Original/ScienceQA/test.json \
        --image-folder data/datasets \
                --mm-text-select-layer -1 \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --single-pred-prompt \
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


python -m llava.eval.Eval.eval_science_qa \
    --base-dir data/instructions/Instructions_Original/ScienceQA \
    --result-file $output_file \
    --output-file $RESULT_DIR/$STAGE/output.jsonl \
    --output-result $RESULT_DIR/$STAGE/output_result.jsonl \
    --output-dir $RESULT_DIR/$STAGE
