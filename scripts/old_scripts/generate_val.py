import itertools
import os
import glob
from shutil import copyfile
import re

def multiple_replace(string, rep_dict):
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)

#####CHANGE- file to source imgs and labels from
val_lbls_dir = "/scratch/public/MW_images/test_cropped_annotated/"
all_val_lbls = glob.glob(val_lbls_dir + "*.txt")

reg_dict = {
    "test_cropped_annotated": "0m_test_cropped",
    ".txt": ".jpg"
}

all_val_imgs = [multiple_replace(x, reg_dict) for x in all_val_lbls]


#Convetional names
img_fname = "val_img_paths.txt"
lbl_fname = "val_lbl_paths.txt"

#Write new domain to within
mw_within_dir = "/home/fcw/repro_bass_300/domain_experiment/BC_team_domain_experiment/Train MW Val MW 100 real 75 syn/baseline/"
if not os.path.exists(mw_within_dir):
    os.makedirs(mw_within_dir)

with open(mw_within_dir + img_fname, "w") as f:
    f.writelines(img + '\n' for img in all_val_imgs)
with open(mw_within_dir + lbl_fname, "w") as f:
    f.writelines(lbl + '\n' for lbl in all_val_lbls)

#Only combinations with MW in them
domains = ["EM", "NE", "NW", "SW", "MW"]
combinations = list(itertools.product(domains, repeat=2))

def containsMW(element):
    return "MW" in element

#Gets combinations with MW in it
MW_combos = list(filter(containsMW, combinations))

for combo in MW_combos:
    if combo[0] == "MW" and combo[1]=="MW":
        continue
    src_dir =  "/home/fcw/repro_bass_300/domain_experiment/BC_team_domain_experiment/Train {val} Val {val} 100 real 75 syn/baseline/".format(train=combo[0],val=combo[1])
    out_dir = "/home/fcw/repro_bass_300/domain_experiment/BC_team_domain_experiment/Train {train} Val {val} 100 real 75 syn/baseline/".format(train=combo[0],val=combo[1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    print(out_dir + img_fname)
    
    copyfile(src_dir + img_fname, out_dir + img_fname)
    copyfile(src_dir + lbl_fname, out_dir + lbl_fname)
