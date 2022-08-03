#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="/scratch/cek28/repro_bass/yolov3/:$PYTHONPATH"

python efficient_run_train_test.py --device 2 --supplemental_batch_size 1 --val_path /scratch/cek28/jitter/wt/experiments/Test_Gray_World_Domain/ --experiment Gray_World
python efficient_run_train_test.py --device 2 --supplemental_batch_size 1 --val_path /scratch/cek28/jitter/wt/experiments/Test_Color_Equalize_Domain/ --experiment Color_Equalize_Domain
python efficient_run_train_test.py --device 2 --supplemental_batch_size 5 --experiment Optimal_Ratio_5
python efficient_run_train_test.py --device 2 --supplemental_batch_size 6 --experiment Optimal_Ratio_6