"""
Runs training script across all pairs of domains among `domains`
"""
import os

# domains = ['EM', 'NE', 'NW', 'SW']

batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8]

for batch_size in batch_sizes:
        # os.system(f'python train.py --source {s_domain} --target {t_domain}')
        s =f"""#!/bin/bash
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yl708@duke.edu     # Where to send mail 
#SBATCH --partition=scavenger-gpu
#SBATCH --exclusive
#SBATCH --time='7-0'
#SBATCH --chdir='/work/yl708/bass/repro_bass/yolov3/'
#SBATCH --mem=0
#SBATCH --output=/work/yl708/bass/repro_bass_files/jitter/wt/experiment_results/Optimal_Ratio_{batch_size}/R-%x.%j.out
#SBATCH --error=/work/yl708/bass/repro_bass_files/jitter/wt/experiment_results/Optimal_Ratio_{batch_size}/R-%x.%j.err

source ~/.bashrc
source ~/.bash_profile
cd /work/yl708/bass/repro_bass/yolov3/

date
hostname

rm /scratch/public
ln -s /work/yl708/bass/repro_bass_files/ /scratch/public

conda activate torch

"""

        s += f'python efficient_run_train_test.py --experiment Optimal_Ratio_{batch_size}_CG --experiment_name Optimal_Ratio_{batch_size} --device 0 --supplemental_batch_size {batch_size}\n'

        with open(f'training_scripts/mb_{batch_size}.sh', 'w') as script:
            script.write(s)
