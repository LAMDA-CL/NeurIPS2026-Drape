#!/bin/bash

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="0,1,2,3"
fi
export CUDA_VISIBLE_DEVICES

bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_vizwiz.sh task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_iconqa.sh task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_clevr.sh task6 ./checkpoints/llava_drape_UCIT_Task6

bash  ./scripts/drape/Eval_UCIT/eval_flickr30k.sh task6 ./checkpoints/llava_drape_UCIT_Task6


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh task5 ./checkpoints/llava_drape_UCIT_Task5

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh task5 ./checkpoints/llava_drape_UCIT_Task5

bash  ./scripts/drape/Eval_UCIT/eval_vizwiz.sh task5 ./checkpoints/llava_drape_UCIT_Task5

bash  ./scripts/drape/Eval_UCIT/eval_iconqa.sh task5 ./checkpoints/llava_drape_UCIT_Task5

bash  ./scripts/drape/Eval_UCIT/eval_clevr.sh task5 ./checkpoints/llava_drape_UCIT_Task5


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh task4 ./checkpoints/llava_drape_UCIT_Task4

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh task4 ./checkpoints/llava_drape_UCIT_Task4

bash  ./scripts/drape/Eval_UCIT/eval_vizwiz.sh task4 ./checkpoints/llava_drape_UCIT_Task4

bash  ./scripts/drape/Eval_UCIT/eval_iconqa.sh task4 ./checkpoints/llava_drape_UCIT_Task4


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh task3 ./checkpoints/llava_drape_UCIT_Task3

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh task3 ./checkpoints/llava_drape_UCIT_Task3

bash  ./scripts/drape/Eval_UCIT/eval_vizwiz.sh task3 ./checkpoints/llava_drape_UCIT_Task3


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh task2 ./checkpoints/llava_drape_UCIT_Task2

bash  ./scripts/drape/Eval_UCIT/eval_arxivqa.sh task2 ./checkpoints/llava_drape_UCIT_Task2


bash  ./scripts/drape/Eval_UCIT/eval_imagenet.sh task1 ./checkpoints/llava_drape_UCIT_Task1