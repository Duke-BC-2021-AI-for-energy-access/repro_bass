from os.path import exists
import glob

def find_nes(file_holders):
    files_dne = set()

    for file_holder in file_holders:
        with open(file_holder, "r") as f:
            files = f.read().split("\n")
        
        for file in files:
            if not exists(file):
                files_dne.add(file)

    return files_dne

val_directory = "/home/fcw/repro_bass_300/domain_experiment/BC_team_domain_experiment/"
val_file_holders = glob.glob(val_directory + "**/baseline/val*.txt", recursive=True)
    
val_nes = find_nes(val_file_holders)

val_file_out = "/home/fcw/scripts/nes/val_nes_fixed.txt"


with open(val_file_out, "w") as f:
   for file in val_nes:
       f.write(file + "\n")

reg_directory = "/home/fcw/updated_txt_files_2/"
reg_file_holders = glob.glob(reg_directory + "*.txt")

reg_nes = find_nes(reg_file_holders)

reg_file_out = "/home/fcw/scripts/nes/reg_nes_fixed.txt"

with open(reg_file_out, "w") as f:
   for file in reg_nes:
       f.write(file + "\n")
