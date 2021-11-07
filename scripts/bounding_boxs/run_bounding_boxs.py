import cv2
import random
import glob
import os
import re
from test_bounding_boxs import create_boxes

#Folder that holds images and txt files
my_txt_dir = "/scratch/public/MW_images/test_cropped_annotated/"
my_img_dir = "/scratch/public/MW_images/0m_test_cropped/"

#Output directory
results_dir = "/scratch/public/scripts/bounding_boxs/bbox_test2/"

create_boxes(my_txt_dir, my_img_dir, results_dir)