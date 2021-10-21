import glob
import os
import re

#Get all txt files from domain txt files 2
directory = "/scratch/dataplus2021/repro_bass_300/domain_txt_files2/"
all_files = glob.glob(directory + "*.txt")

#Creates directory to save txt files to (since you can't save in scratch)
save_directory = "/home/fcw/updated_txt_files_2/"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print(save_directory + " directory was made")

def multiple_replace(string, rep_dict):
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)

reg_dict = {
    "/hdd/dataplus2021/fcw/": "/scratch/dataplus2021/",
    "/hdd/dataplus2021/share/domain_experiment/BC_team_domain_experiment/data/": "/home/fcw/repro_bass_300/domain_experiment/BC_team_domain_experiment/data/"
}

for file in all_files:
    #Gets file name without path (e.g. train_NW_val_SW_imgs.txt)
    current_fname = file[file.rfind("/")+1:]
    print(current_fname)
    #Reads in txt of current file
    txt = open(file, "r").read()
    #Replaces file path
    new_txt = multiple_replace(txt, reg_dict)
    
    #Creates new file path using directory to save to and current file name
    new_fname = save_directory + current_fname
    #Opens file and saves txt with replaced file paths to opened file
    new_file = open(new_fname, "w")
    new_file.write(new_txt)

