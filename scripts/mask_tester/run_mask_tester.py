import glob
import cv2
import test_masks

#Directory holding synthetic images
my_dir = "/hdd/dataplus2021/fcw/ImageAugment4/results9"

#Results directory
res_dir = "/hdd/dataplus2021/fcw/MaskTester/results/"

test_all_masks(my_dir, res_dir)