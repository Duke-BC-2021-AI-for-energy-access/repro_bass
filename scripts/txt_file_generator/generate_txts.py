#Need 100 from each domain
#Need 75 synthetic from each domain
import random
import itertools
import os
import glob
import argparse
from generate_txt_functions import readDomains
from generate_txt_functions import *
from shutil import copyfile

parser = argparse.ArgumentParser(description='YOLO txt file generator for imgs and lbls')

parser.add_argument('--domains_file', type=str, default='/scratch/public/scripts/domains.txt', help='file holding domains to be used')
parser.add_argument('--txt_files_dir', type=str, default='/home/fcw/updated_txt_files_2/', help='directory holding txt files')
parser.add_argument('--results_dir', type=str, default='/scratch/public/txt_files_test/', help='directory to output txt files to')

parser.add_argument('--new_domain', type=str, default='MW', help='new domain to add')
parser.add_argument('--new_domain_real_dir', type=str, default='/scratch/public/MW_images/0m_train_cropped/', help='new domain to add')
parser.add_argument('--new_domain_synth_dir', type=str, default='/scratch/public/experimental_output_ratio_MW/', help='new domain to add')

parser.add_argument('--new_synth_dir', type=str, default='/scratch/public/unused/', help='directory holding 75 extra images to sample (each domain)')


args = parser.parse_args()

txt_files_dir = args.txt_files_dir
###NEED TO CHANGE TO EXPERIMENT
#pass in experiment directory (e.g. /scratch/public/frankie_repro_bass/domain_experiment/BC_team_domain_experiment)
results_dir = args.results_dir
domain = args.new_domain
domain_real_dir = args.new_domain_real_dir
domain_synth_dir = args.new_domain_synth_dir

new_synth_dir = args.new_synth_dir

with open(args.domains_file, "r") as f:
    domains = f.read().strip().split(",")

random.seed(42)

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

#Gives my home directory
#os.path.expanduser(f"~{pwd.getpwuid(os.geteuid())[0]}/")

real_imgs, real_lbls, synth_imgs, synth_lbls = readDomains(domains, txt_files_dir, new_synth_dir)

val_dir = "../domain_experiment/BC_team_domain_experiment/"
val_imgs, val_lbls = readVals(domains, val_dir)

combinations = list(itertools.product(domains, repeat=2))

#Need to add functionality for addVal when you need to add validation
if domain is not None:
    addDomain(domain, domains, domain_real_dir, domain_synth_dir, real_imgs, synth_imgs, real_lbls, synth_lbls, 100, 75, val_domain)

    new_all_combos = list(itertools.product(domains, repeat=2))
    combinations = list(filter(lambda elem: containsNewDomain(domain, elem),new_all_combos))
    #combinations = list(filter(containsNewDomain, list(itertools.product(domains, repeat=2))))

for combo in combinations:
    real = combo[0]
    synth = combo[1]

    create_a_file(real, synth, real_imgs, synth_imgs, "train", "img", results_dir)
    create_a_file(real, synth, real_lbls, synth_lbls, "train", "lbl", results_dir)
    
    create_a_file(real, synth, val_imgs, [], "val", "img", results_dir)
    create_a_file(real, synth, val_lbls, [], "val", "lbl", results_dir)
#for combo in combinations:
