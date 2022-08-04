#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/scratch/cek28/repro_bass/yolov3/:$PYTHONPATH"

python efficient_run_train_test.py --device 0 --supplemental_batch_size 0 --experiment Baseline
python efficient_run_train_test.py --device 0 --supplemental_batch_size 1 --experiment Lower_Bound
python efficient_run_train_test.py --device 0 --supplemental_batch_size 6 --experiment Optimal_Ratio_6
python efficient_run_train_test.py --device 0 --supplemental_batch_size 7 --experiment Optimal_Ratio_7