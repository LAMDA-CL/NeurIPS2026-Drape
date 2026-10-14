#!/bin/bash

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="0,1,2,3"
fi
export CUDA_VISIBLE_DEVICES

bash scripts/drape/Eval_CoIN/1_ScienceQA.sh task1 ./checkpoints/llava_drape_CoIN_Task1

bash scripts/drape/Eval_CoIN/1_ScienceQA.sh task2 ./checkpoints/llava_drape_CoIN_Task2
bash scripts/drape/Eval_CoIN/2_TextVQA.sh task2 ./checkpoints/llava_drape_CoIN_Task2

bash scripts/drape/Eval_CoIN/1_ScienceQA.sh task3 ./checkpoints/llava_drape_CoIN_Task3
bash scripts/drape/Eval_CoIN/2_TextVQA.sh task3 ./checkpoints/llava_drape_CoIN_Task3
bash scripts/drape/Eval_CoIN/3_ImageNet.sh task3 ./checkpoints/llava_drape_CoIN_Task3

bash scripts/drape/Eval_CoIN/1_ScienceQA.sh task4 ./checkpoints/llava_drape_CoIN_Task4
bash scripts/drape/Eval_CoIN/2_TextVQA.sh task4 ./checkpoints/llava_drape_CoIN_Task4
bash scripts/drape/Eval_CoIN/3_ImageNet.sh task4 ./checkpoints/llava_drape_CoIN_Task4
bash scripts/drape/Eval_CoIN/4_GQA.sh task4 ./checkpoints/llava_drape_CoIN_Task4

bash scripts/drape/Eval_CoIN/1_ScienceQA.sh task5 ./checkpoints/llava_drape_CoIN_Task5
bash scripts/drape/Eval_CoIN/2_TextVQA.sh task5 ./checkpoints/llava_drape_CoIN_Task5
bash scripts/drape/Eval_CoIN/3_ImageNet.sh task5 ./checkpoints/llava_drape_CoIN_Task5
bash scripts/drape/Eval_CoIN/4_GQA.sh task5 ./checkpoints/llava_drape_CoIN_Task5
bash scripts/drape/Eval_CoIN/5_VizWiz.sh task5 ./checkpoints/llava_drape_CoIN_Task5

bash scripts/drape/Eval_CoIN/1_ScienceQA.sh task6 ./checkpoints/llava_drape_CoIN_Task6
bash scripts/drape/Eval_CoIN/2_TextVQA.sh task6 ./checkpoints/llava_drape_CoIN_Task6
bash scripts/drape/Eval_CoIN/3_ImageNet.sh task6 ./checkpoints/llava_drape_CoIN_Task6
bash scripts/drape/Eval_CoIN/4_GQA.sh task6 ./checkpoints/llava_drape_CoIN_Task6
bash scripts/drape/Eval_CoIN/5_VizWiz.sh task6 ./checkpoints/llava_drape_CoIN_Task6
bash scripts/drape/Eval_CoIN/6_Grounding.sh task6 ./checkpoints/llava_drape_CoIN_Task6

bash scripts/drape/Eval_CoIN/1_ScienceQA.sh task7 ./checkpoints/llava_drape_CoIN_Task7
bash scripts/drape/Eval_CoIN/2_TextVQA.sh task7 ./checkpoints/llava_drape_CoIN_Task7
bash scripts/drape/Eval_CoIN/3_ImageNet.sh task7 ./checkpoints/llava_drape_CoIN_Task7
bash scripts/drape/Eval_CoIN/4_GQA.sh task7 ./checkpoints/llava_drape_CoIN_Task7
bash scripts/drape/Eval_CoIN/5_VizWiz.sh task7 ./checkpoints/llava_drape_CoIN_Task7
bash scripts/drape/Eval_CoIN/6_Grounding.sh task7 ./checkpoints/llava_drape_CoIN_Task7
bash scripts/drape/Eval_CoIN/7_VQAv2.sh task7 ./checkpoints/llava_drape_CoIN_Task7

bash scripts/drape/Eval_CoIN/1_ScienceQA.sh task8 ./checkpoints/llava_drape_CoIN_Task8
bash scripts/drape/Eval_CoIN/2_TextVQA.sh task8 ./checkpoints/llava_drape_CoIN_Task8
bash scripts/drape/Eval_CoIN/3_ImageNet.sh task8 ./checkpoints/llava_drape_CoIN_Task8
bash scripts/drape/Eval_CoIN/4_GQA.sh task8 ./checkpoints/llava_drape_CoIN_Task8
bash scripts/drape/Eval_CoIN/5_VizWiz.sh task8 ./checkpoints/llava_drape_CoIN_Task8
bash scripts/drape/Eval_CoIN/6_Grounding.sh task8 ./checkpoints/llava_drape_CoIN_Task8
bash scripts/drape/Eval_CoIN/7_VQAv2.sh task8 ./checkpoints/llava_drape_CoIN_Task8
bash scripts/drape/Eval_CoIN/8_OCRVQA.sh task8 ./checkpoints/llava_drape_CoIN_Task8
