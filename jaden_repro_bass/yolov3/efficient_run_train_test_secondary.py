from dataset import Dataset
import os
import subprocess
import itertools

#CHANGE
out_path = '/scratch/public/MW_batch_size_8_secondary_out/'
# out_path = '/scratch/public/_experimental_outputs/'
train_path = '/scratch/public/jaden_repro_bass/MW_txt_files/'
val_path = '/scratch/public/jaden_repro_bass/domain_experiment/BC_team_domain_experiment/'

domains = ["EM", "NE", "NW", "SW", "MW"]

combinations = list(itertools.product(domains, repeat=2))

def containsMW(element):
    return "MW" in element[0] and "MW" not in element[1]

#Gets combinations with MW in it
MW_combos = list(filter(containsMW, combinations))

datasets = []
for combo in MW_combos:
  for i in range(0,4):
    dataset_string = """Dataset(img_txt=train_path+'train_{src}_val_{dst}_imgs.txt',
                      lbl_txt=train_path+'train_{src}_val_{dst}_lbls.txt',
                      out_dir=out_path+'t_{src}_v_{dst}_{i}/',
                      img_txt_val=val_path+'Train {src} Val {dst} 100 real 75 syn/baseline/val_img_paths.txt',
                      lbl_txt_val=val_path+'Train {src} Val {dst} 100 real 75 syn/baseline/val_lbl_paths.txt')""".format(src=combo[0],dst=combo[1],i=i)
    datasets.append(eval(dataset_string))

#Could create some variable that does not use every trial

for trial in datasets:
    subprocess.run(['python', 'run_save_train_test.py',
                    '--img_list', trial.get_img_txt(), 
                    '--lbl_list', trial.get_lbl_txt(),
                    '--epochs', '300',
                    '--out_dir', trial.get_out_dir(),
                    '--img_list_val', trial.get_img_txt_val(), 
                    '--lbl_list_val', trial.get_lbl_txt_val(),
                    '--version', 'v2'])