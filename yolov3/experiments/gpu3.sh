#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="/scratch/cek28/repro_bass/yolov3/:$PYTHONPATH"

python ../efficient_run_train_test.py --device 3 --supplemental_batch_size 1 --experiment Cyclegan
python ../efficient_run_train_test.py --device 3 --supplemental_batch_size 4 --experiment Optimal_Ratio_4
python ../efficient_run_train_test.py --device 3 --supplemental_batch_size 5 --experiment Optimal_Ratio_5