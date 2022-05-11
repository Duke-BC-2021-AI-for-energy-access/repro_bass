import itertools
import os
from shutil import copyfile

#Only combinations with MW in them
domains = ["EM", "NE", "NW", "SW", "MW"]
combinations = list(itertools.product(domains, repeat=2))

def containsMW(element):
    return "MW" in element

#Gets combinations with MW in it
MW_combos = list(filter(containsMW, combinations))

img_shapes = "/scratch/cek28/jaden_repro_bass/domain_experiment/BC_team_domain_experiment/Train EM Val EM 100 real 75 syn/baseline/val_img_paths.shapes"

img_fname = "val_img_paths.shapes"
lbl_fname = "val_lbl_paths.shapes"

for combo in MW_combos:
    out_dir = "/scratch/cek28/jaden_repro_bass/domain_experiment/BC_team_domain_experiment/Train {train} Val {val} 100 real 75 syn/baseline/".format(train=combo[0],val=combo[1])
    
    print(out_dir + img_fname)
    
    copyfile(img_shapes, out_dir + img_fname)
    copyfile(img_shapes, out_dir + lbl_fname)
