#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="/scratch/cek28/repro_bass/yolov3/:$PYTHONPATH"

python ../efficient_run_train_test.py --device 5 --supplemental_batch_size 1 --experiment Histogram_Matching
python ../efficient_run_train_test.py --device 5 --supplemental_batch_size 1 --experiment Color_Equalize_Domain
python ../efficient_run_train_test.py --device 5 --supplemental_batch_size 8 --experiment Optimal_Ratio_8