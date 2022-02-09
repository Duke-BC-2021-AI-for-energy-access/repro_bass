from dataset import Dataset
import os
import subprocess
import itertools
import argparse
import pwd

repo_path = os.path.expanduser(f"~{pwd.getpwuid(os.geteuid())[0]}/") + 'repro_bass/'

parser = argparse.ArgumentParser()

#out_path- /scratch/public/results/25background_experiment/
parser.add_argument('--out_path', default=os.path.expanduser(f"~{pwd.getpwuid(os.geteuid())[0]}/")+'MW_batch_size_8/')
#train_path- /scratch/public/txt_files/25background_experiment/
parser.add_argument('--train_path', default='/scratch/public/txt_files/cyclegan_txt_files/')
parser.add_argument('--val_path', default='/scratch/public/domain_experiment/BC_team_domain_experiment/')
parser.add_argument('--epochs', default='300')
parser.add_argument('--device', default='0')
args = parser.parse_args()

out_path = args.out_path
train_path = args.train_path
val_path = args.val_path

domains = ["EM", "NE", "NW", "SW"]

combinations = list(itertools.product(domains, repeat=2))

#Adds filter as necessary

#def containsMW(element):
    #element tuple of (train, val)
#    return "MW" in element[0] and "MW" not in element[1]

def containsDuplicate(element):
  return element[0] != element[1]

#Gets combinations with MW in it
#MW_combos = list(filter(containsMW, combinations))

reg_combos = list(filter(containsDuplicate, combinations))

datasets = []
for combo in reg_combos:
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
                    '--epochs', args.epochs,
                    '--out_dir', trial.get_out_dir(),
                    '--img_list_val', trial.get_img_txt_val(),
                    '--lbl_list_val', trial.get_lbl_txt_val(),
                    '--version', 'v2',
                    '--device', args.device])
