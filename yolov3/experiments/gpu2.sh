#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="/scratch/cek28/repro_bass/yolov3/:$PYTHONPATH"

python ../efficient_run_train_test.py --device 2 --supplemental_batch_size 1 --experiment Optimal_Ratio_1
python ../efficient_run_train_test.py --device 2 --supplemental_batch_size 2 --experiment Optimal_Ratio_2
python ../efficient_run_train_test.py --device 2 --supplemental_batch_size 3 --experiment Optimal_Ratio_3