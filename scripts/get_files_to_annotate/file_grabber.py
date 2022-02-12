from os.path import exists
from shutil import copyfile
import os
import get_files_annotate

files_arr = open("NW_train.txt", "r").read().replace('\'', '').split("\n")

train_dir = "/scratch/public/images/jitter/NW_train/"
#test_dir = "/scratch/public/MW_images/0m_test_cropped/"

results_dir = "/home/fcw/NW_training_images/"
        
get_files(files_arr, train_dir, results_dir)