import glob
import itertools

#For each domain:
#Glob.glob recursively, grabbing GP GAN files
#Collect the file names (can rfind the / or split by / and take last)
#For each file name:
#Find in either (check associated using dict)
#/scratch/dataplus2021/all_imagery/domain_name/images/
#/scratch/dataplus2021/repro_bass_300/domain_experiment/BC_team_domain_experiment/background_by_domain/



domains = ["EM", "NW", "NE", "SW", "MW"]

real_imgs = dict()
real_lbls = dict()
background_dict = dict()

output_dir = ""

txt_files_dir = "/home/fcw/updated_txt_files_2/"
background_dir = "/scratch/dataplus2021/repro_bass_300/domain_experiment/BC_team_domain_experiment/background_by_domain/"
combinations = list(itertools.product(domains, repeat=2))

for domain in domains:
    within_img_file = txt_files_dir + "train_{domain}_val_{domain}_imgs.txt".format(domain=domain)
    with open(within_img_file, "r") as f:
        file_txt = list(filter(None, f.read().split("\n")))
        real_imgs[domain] = file_txt[:100]
        real_lbls[domain]  = [x.replace(".jpg", ".txt") for x in real_imgs[domain]]
    
    background_files = glob.glob(background_dir + "{domain}/*.jpg".format(domain=domain), recursive=True)
    
#MW case
