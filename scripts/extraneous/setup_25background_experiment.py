import glob
import os
import re

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

reg_dict = {
        ".jpg": ".txt",
        "images": "labels"
}


#os.mkdir('/scratch/public/txt_files/25background_experiment/')

#empty_folder = "/scratch/public/txt_files/empty_txt_files/"
#os.mkdir(empty_folder)

hdd_string_to_replace = "/hdd/dataplus2021/share/"
replacer_string = "/scratch/public/"

input_directories = ["/scratch/public/txt_files/domain_txt_files2/", "/scratch/public/txt_files/MW_txt_files/"]

#background_dir = "/scratch/dataplus2021/repro_bass_300/domain_experiment/BC_team_domain_experiment/background_by_domain/"
background_dir = "/scratch/public/domain_experiment/BC_team_domain_experiment/background_by_domain/"
MW_background_dir = "/scratch/public/MW_images/train_background_cropped/"


output_dir = "/scratch/public/txt_files/25background_experiment/"

img_txt_files = []

for dir in input_directories:
    img_txt_files.extend(glob.glob(dir + "*imgs.txt"))

for file in img_txt_files:
    with open(file, "r") as f:
        lines = f.read().split('\n')
        baseline_images = [x.replace(hdd_string_to_replace, replacer_string) + '\n' for x in lines if "ratio" not in x and x != '']
        background_images = [x for x in lines if "ratio" in x and x != '']
        #print(background_images)
    val_domain = file.split('val_')[1][0:2]
    domain_background_dir = background_dir + val_domain
    if val_domain == "MW":
        domain_background_dir = MW_background_dir
    
    background_img_names = set([domain_background_dir + '/' + bg_img[bg_img.rfind("/")+1:] + '\n' for bg_img in background_images])
    
    output_img_file = output_dir + file[file.rfind("/")+1:]
    with open(output_img_file, "w") as f:
        f.writelines(baseline_images)
        f.writelines(background_img_names)
    
    output_lbl_file = output_img_file.replace('img', 'lbl')
    baseline_lbls = [multiple_replace(reg_dict, x) for x in baseline_images]
    background_lbl_names  = [x.replace('.jpg', '.txt') for x in background_img_names]
    
    #background_lbl_names = [empty_folder + bg_img[bg_img.rfind("/")+1:].replace('.jpg', '.txt') for bg_img in background_img_names]

    for lbl in background_lbl_names:
        with open(lbl, "w") as f:
            pass

    with open(output_lbl_file, "w") as f:
        f.writelines(baseline_lbls)
        f.writelines(background_lbl_names)