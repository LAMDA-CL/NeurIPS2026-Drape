#!/bin/bash

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="0,1,2,3"
fi
export CUDA_VISIBLE_DEVICES

bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh hide-task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh hide-task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_vizwiz.sh hide-task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_iconqa.sh hide-task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_clevr.sh hide-task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_flickr30k.sh hide-task6 ./checkpoints/llava_drape_UCIT_Task6


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh hide-task5 ./checkpoints/llava_drape_UCIT_Task5

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh hide-task5 ./checkpoints/llava_drape_UCIT_Task5

bash  ./scripts/drape/Eval_UCIT/eval_vizwiz.sh hide-task5 ./checkpoints/llava_drape_UCIT_Task5

bash  ./scripts/drape/Eval_UCIT/eval_iconqa.sh hide-task5 ./checkpoints/llava_drape_UCIT_Task5

bash  ./scripts/drape/Eval_UCIT/eval_clevr.sh hide-task5 ./checkpoints/llava_drape_UCIT_Task5


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh hide-task4 ./checkpoints/llava_drape_UCIT_Task4

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh hide-task4 ./checkpoints/llava_drape_UCIT_Task4

bash  ./scripts/drape/Eval_UCIT/eval_vizwiz.sh hide-task4 ./checkpoints/llava_drape_UCIT_Task4

bash  ./scripts/drape/Eval_UCIT/eval_iconqa.sh hide-task4 ./checkpoints/llava_drape_UCIT_Task4


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh hide-task3 ./checkpoints/llava_drape_UCIT_Task3

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh hide-task3 ./checkpoints/llava_drape_UCIT_Task3

bash  ./scripts/drape/Eval_UCIT/eval_vizwiz.sh hide-task3 ./checkpoints/llava_drape_UCIT_Task3


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh hide-task2 ./checkpoints/llava_drape_UCIT_Task2

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh hide-task2 ./checkpoints/llava_drape_UCIT_Task2


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh hide-task1 ./checkpoints/llava_drape_UCIT_Task1