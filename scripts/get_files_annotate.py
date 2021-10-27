from os.path import exists
from shutil import copyfile
import os

files_arr = open("MW_annotate.txt", "r").read().split("\n")

train_dir = "/scratch/public/MW_images/0m_train_cropped/"
test_dir = "/scratch/public/MW_images/0m_test_cropped/"
results_dir = "/home/fcw/scripts/MW_annotate/"

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

for file in files_arr:
    if exists(train_dir + file):
        copyfile(train_dir + file, results_dir + file)
    elif exists(test_dir + file):
        copyfile(test_dir + file, results_dir + file)
        