#!/bin/bash

if [ -z "${GPUS:-}" ]; then
    GPUS="0,1,2,3"
fi
export GPUS

bash scripts/drape/Train_CoIN/Task1.sh

bash scripts/drape/Train_CoIN/Task2.sh

bash scripts/drape/Train_CoIN/Task3.sh

bash scripts/drape/Train_CoIN/Task4.sh

bash scripts/drape/Train_CoIN/Task5.sh

bash scripts/drape/Train_CoIN/Task6.sh

bash scripts/drape/Train_CoIN/Task7.sh

bash scripts/drape/Train_CoIN/Task8.sh

