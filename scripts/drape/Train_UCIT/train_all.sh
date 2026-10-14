#!/bin/bash

if [ -z "${GPUS:-}" ]; then
    GPUS="0,1,2,3"
fi
export GPUS

bash scripts/drape/Train_UCIT/Task1.sh

bash scripts/drape/Train_UCIT/Task2.sh

bash scripts/drape/Train_UCIT/Task3.sh

bash scripts/drape/Train_UCIT/Task4.sh

bash scripts/drape/Train_UCIT/Task5.sh

bash scripts/drape/Train_UCIT/Task6.sh
