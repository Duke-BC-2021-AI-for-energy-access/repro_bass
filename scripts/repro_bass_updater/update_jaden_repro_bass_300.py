import glob
import os
import re

#Get all txt files from various train folders
#Ensure val files as those are only ones to change (improve efficiency)

def update(home_repro):
    """
    Updates repro bass folder to have file names that correspond to home directory,
    overcomes image reading permission errors

    Args:
        home_repro ([type]): Repo to change file names to be in respect to

    Returns:
        [type]: [description]
    """

    reg_dict = {
        "/scratch/public/jaden_repro_bass/": home_repro
    }

    directory = home_repro + "domain_experiment/BC_team_domain_experiment/"

    #Change validation files
    all_files = glob.glob(directory + "**/baseline/val*.txt", recursive=True)

    #Change YOLO scripts
    all_files.append(home_repro + "yolov3/efficient_run_train_test.py")
    all_files.append(home_repro + "yolov3/run_save_train_test.py")

    #Change MW train img and lbl files
    MW_txt_directory = home_repro + "MW_txt_files/"

    for x in glob.glob(MW_txt_directory + "*.txt"):
        all_files.append(x)

    #Creates directory to save txt files to (since you can't save in scratch)
    #save_directory = "/home/fcw/updated_txt_files/"
    #if not os.path.exists(save_directory):
    #    os.makedirs(save_directory)
    #    print(save_directory + " directory was made")


    def multiple_replace(string, rep_dict):
        pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
        return pattern.sub(lambda x: rep_dict[x.group(0)], string)

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