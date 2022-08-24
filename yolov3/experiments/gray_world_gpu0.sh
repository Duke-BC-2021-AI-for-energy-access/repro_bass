#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/scratch/cek28/repro_bass/yolov3/:$PYTHONPATH"

python ../efficient_run_train_test.py --device 0 --supplemental_batch_size 1 --experiment Gray_World