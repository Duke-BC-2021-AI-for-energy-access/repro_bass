import glob
from os.path import exists
import os
from shutil import copyfile

domains = ["EM", "NE", "NW", "SW"]
train_val_paths = ["Train EM Val EM 100 real 75 syn", "Train NE Val NE 100 real 75 syn", "Train NW Val NW 100 real 75 syn", "Train SW Val SW 100 real 75 syn"]

upper_directory = "/scratch/dataplus2021/all_imagery/" 
all_imagery_files = glob.glob(upper_directory+"*.jpg")

train_val_directory = "/scratch/dataplus2021/repro_bass_300/domain_experiment/BC_team_domain_experiment/"

results_dir = "/home/yaz/unused/"

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

for i in range(len(domains)):
    actual_dir = upper_directory + domains[i] + "/images/"
    image_files = glob.glob(actual_dir + "*")

    if not os.path.exists(results_dir + domains[i]):
        os.mkdir(results_dir + domains[i])
    
    train_txt = train_val_directory + train_val_paths[i] + "/baseline/training_img_paths.txt"
    val_txt = train_val_directory + train_val_paths[i] + "/baseline/val_img_paths.txt"
    
    train_file = open(train_txt, "r")
    val_file = open(val_txt, "r")
    train_file = train_file.read()
    val_file = val_file.read()

    for image in image_files:
        image_name = image.split("/")[-1]
        if image_name not in train_file and image_name not in val_file:
            copyfile(image, results_dir + domains[i] + "/" + image_name)
