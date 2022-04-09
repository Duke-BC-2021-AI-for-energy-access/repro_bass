import cv2
import random
import glob
import os
import re
from test_bounding_boxs import create_boxes

#Folder that holds images and txt files
#my_txt_dir = "/scratch/public/new_cropped_turbines/labels/"
#my_img_dir = "/scratch/public/new_cropped_turbines/images/"
my_txt_dir = "/scratch/public/augmented_images/EM/labels/"
my_img_dir = "/scratch/public/augmented_images/EM/canvases/"

#Images and Labels txt file
#imgs = "/scratch/public/jitter/wt/experiments/Optimal_Ratio_1_CG/Train_EM_Test_SW_Supplement_Images.txt"
#txts = "/scratch/public/jitter/wt/experiments/Optimal_Ratio_1_CG/Train_EM_Test_SW_Supplement_Labels.txt"

#Output directory
#results_dir = "/home/fcw/relative_boxes_test/"
results_dir = "/home/fcw/augmented_test/"

create_boxes(my_txt_dir, my_img_dir, results_dir, False)