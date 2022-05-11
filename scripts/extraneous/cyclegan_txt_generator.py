import glob
import os
import itertools
import re
import random

random.seed(42)

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 


cycle_dir = "/scratch/cek28/cyclegan-output/"

txt_files_dir = "/home/fcw/updated_txt_files/"

output_dir = "/scratch/cek28/txt_files/cyclegan_txt_files/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


reg_dict = {
    ".jpg": ".txt",
    "images": "labels"

}

domains = ["EM", "NE", "SW", "NW"]

real_imgs = {}
real_lbls = {}

for src in domains:

    txt_img_file = txt_files_dir + "train_{domain}_val_{domain}_imgs.txt".format(domain=src)

    with open(txt_img_file, "r") as f:
        file_txt = list(filter(None, f.read().split("\n")))
        real_imgs[src] = file_txt[:100]
        real_lbls[src]  = [multiple_replace(reg_dict, x) for x in real_imgs[src]]

    for target in domains:
        if src == target:
            continue

        subfolder = "/src_{src}_targ_{targ}/".format(src=src,targ=target)

        cycle_img_folder = cycle_dir + "images" + subfolder
        cycle_lbl_folder = cycle_img_folder.replace("images", "labels")

        cycle_imgs = glob.glob(cycle_img_folder + "*.jpg")
        cycle_lbls = glob.glob(cycle_img_folder + "*.txt")

        cycle_imgs_used = random.sample(cycle_imgs, 75)
        cycle_lbls_used = [multiple_replace(reg_dict, x) for x in cycle_imgs_used]

        output_img_file = output_dir + "train_{src}_val_{targ}_imgs.txt".format(src=src,targ=target)
        output_lbl_file = output_img_file.replace("imgs", "lbls")
        output_shapes_file = output_img_file.replace(".txt", ".shapes")

        with open(output_img_file, "w") as f:
            f.writelines("%s\n" % l for l in real_imgs[src])
            f.writelines("%s\n" % l for l in cycle_imgs_used)
        
        with open(output_lbl_file, "w") as f:
            f.writelines("%s\n" % l for l in real_lbls[src])
            f.writelines("%s\n" % l for l in cycle_lbls_used)

        with open(output_shapes_file, "w") as f:
            f.writelines("608\n" for l in real_imgs[src])
            f.writelines("608\n" for l in cycle_imgs_used)
        

        
        
