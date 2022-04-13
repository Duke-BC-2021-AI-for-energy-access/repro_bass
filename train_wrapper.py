"""
Runs training script across all pairs of domains among `domains`
"""
import os

# domains = ['EM', 'NE', 'NW', 'SW']

# batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8]

experiment_names = ["Color_Equalize_Domain", "Color_Equalize_Domain_100", "Lower_Bound_100", "Optimal_Ratio_100_1", "Test_Color_Equalize_Domain"]

# for batch_size in batch_sizes:
for experiment in experiment_names:
        # Create directory for output log file
        os.system(f'mkdir -p /work/yl708/bass/repro_bass_files/jitter/wt/experiment_results/{experiment}/')

        # os.system(f'python train.py --source {s_domain} --target {t_domain}')
        s =f"""#!/bin/bash
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yl708@duke.edu     # Where to send mail 
#SBATCH --partition=scavenger-gpu
#SBATCH --exclusive
#SBATCH --time='7-0'
#SBATCH --chdir='/work/yl708/bass/repro_bass/yolov3/'
#SBATCH --mem=0
#SBATCH --output=/work/yl708/bass/repro_bass_files/jitter/wt/experiment_results/{experiment}/R-%x.%j.out
#SBATCH --error=/work/yl708/bass/repro_bass_files/jitter/wt/experiment_results/{experiment}/R-%x.%j.err

source ~/.bashrc
source ~/.bash_profile
cd /work/yl708/bass/repro_bass/yolov3/

date
hostname

rm /scratch/public
ln -s /work/yl708/bass/repro_bass_files/ /scratch/public

conda activate torch

"""

        s += f'python efficient_run_train_test.py --experiment {experiment} --experiment_name {experiment} --device 0 --supplemental_batch_size 1\n'

        with open(f'training_scripts/mb_{experiment}.sh', 'w') as script:
            script.write(s)
