#Need 100 from each domain
#Need 75 synthetic from each domain
import random
import itertools
import os
import glob

def containsMW(element):
    return "MW" in element

def create_a_file(real, synth, real_files, synth_files, img_or_lbl, results_dir):
    my_real_imgs = real_files[real]
    my_synth_imgs = synth_files[synth]

    fname = "train_{real}_val_{synth}_{img_or_lbl}s.txt".format(real=real, synth=synth, img_or_lbl=img_or_lbl)

    with open(results_dir + fname, "w") as f:
        f.writelines("%s\n" % l for l in my_real_imgs)
        f.writelines("%s\n" % l for l in my_synth_imgs)

###OUTPUT FOLDER- CHANGE
results_dir = "/home/fcw/MW_txt_files/"

#if not os.path.exists(results_dir):
#    os.mkdir(results_dir)

random.seed(42)

####DOMAIN TO SAMPLE FROM
MW_dir = "/scratch/cek28/experimental_output_ratio_MW"
all_MW_imgs = glob.glob(MW_dir + "/**/*.jpg", recursive = True)

####Directory with txt files for within domain of other domains
txt_files_dir = "/home/fcw/updated_txt_files_2/"
real_imgs = dict()
real_lbls = dict()
synth_imgs = dict()
synth_lbls = dict()

#Domains for which we have files
domains = ["EM", "NE", "NW", "SW"]

#Extracts real and synthetic images from txt files
for domain in domains:
    within_img_file = txt_files_dir + "train_{domain}_val_{domain}_imgs.txt".format(domain=domain)
    with open(within_img_file, "r") as f:
        file_txt = list(filter(None, f.read().split("\n")))
        real_imgs[domain] = file_txt[:100]
        synth_imgs[domain] = file_txt[100:]
        #assumes correct naming convention
        real_lbls[domain]  = [x.replace(".jpg", ".txt") for x in real_imgs[domain]]
        synth_lbls[domain]  = [x.replace(".jpg", ".txt") for x in synth_imgs[domain]]

#Adds MW to domains to inlcude all domains
domains.append("MW")
combinations = list(itertools.product(domains, repeat=2))
#Gets combinations with MW in it
MW_combos = list(filter(containsMW, combinations))
#Depends on random seed

#Gets 100 real images
MW_real_dir = "/scratch/cek28/MW_images/0m_train_cropped/"
MW_real_imgs = glob.glob(MW_real_dir + "*.jpg")

#Gets 75 synthetic images
MW_sample_imgs = random.sample(all_MW_imgs, 75)

#Adds MW to dictionary
real_imgs["MW"] = MW_real_imgs
synth_imgs["MW"] = MW_sample_imgs
real_lbls["MW"] = [x.replace(".jpg", ".txt") for x in real_imgs["MW"]]
synth_lbls["MW"] = [x.replace(".jpg", ".txt") for x in synth_imgs["MW"]]

for combo in MW_combos:
    real = combo[0]
    synth = combo[1]

    create_a_file(real, synth, real_imgs, synth_imgs, "img", results_dir)

    create_a_file(real, synth, real_lbls, synth_lbls, "lbl", results_dir)

