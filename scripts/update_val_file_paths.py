import glob
import os
import re

#Get all txt files from various train folders
#Ensure val files as those are only ones to change (improve efficiency)
directory = "/home/fcw/repro_bass_300/domain_experiment/BC_team_domain_experiment/"

all_files = glob.glob(directory + "**/baseline/val*.txt", recursive=True)

#Creates directory to save txt files to (since you can't save in scratch)
#save_directory = "/home/fcw/updated_txt_files/"
#if not os.path.exists(save_directory):
#    os.makedirs(save_directory)
#    print(save_directory + " directory was made")

def multiple_replace(string, rep_dict):
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)

reg_dict = {
    "/hdd/dataplus2021/ak478/": "/home/fcw/"
}

for file in all_files:
    print(file)

    #Reads in txt of current file
    with open(file, "r") as f:
        txt = f.read()
    
    #Replaces file path
    new_txt = multiple_replace(txt, reg_dict)
    
    #Reppens file and saves txt with replaced file paths to opened file
    with open(file, "w") as new_file:
        new_file.write(new_txt)



