import sys
sys.path.insert(1, '/home/fcw/repro_bass/scripts/txt_file_generator/')
from generate_txt_functions import *
import glob
from shutil import copyfile
import os

##ARGUMENTS
domains = ["EM", "MW", "SW", "NW", "NE"]
txt_files_dir = "/home/fcw/updated_txt_files_2/"
output_folder = "/scratch/public/images/"

_, __, synth_imgs, synth_lbls = readDomains(domains, txt_files_dir, None)

file_string = "experimental_output_ratio"

for domain in synth_imgs:
    synth_imgs_in_domain = synth_imgs[domain]
    synth_lbls_in_domain = synth_lbls[domain]
    new_dir = output_folder + domain + "_used_background/"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for image in synth_imgs_in_domain:
        fname = image.rsplit('/', 1)[-1]
        print(fname)
        copyfile(image, new_dir + fname)
        copyfile(image, new_dir + fname)
    #for image in synth_lbls_in_domain:
    #    fname = image.rsplit('/', 1)[-1]
    #    print(fname)
    #    copyfile(image, new_dir + fname)
    #    copyfile(image, new_dir + fname)

"""
MW_files = glob.glob(MW_dir + "*.txt")
for file in MW_files:
    fname = file.rsplit('/', 1)[-1]
    copyfile(file, txt_files_dir + fname)
    copyfile(file, txt_files_dir + fname)
"""