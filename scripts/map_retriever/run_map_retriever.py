import glob
import os
from retrieve_maps import retrieve

#directory holding experimental outputs
directory = "/scratch/public/jitter/wt/experiment_results/Optimal_Ratio_100_8/"

#Will likely stay some- specified in run save train test py
version = "v2"

#Directory and csv file name to output to
# output_dir = "/scratch/public/jitter/wt/experiment_results"
output_dir = directory ## outputting the results to wherever we ran the experiment
output_fname = "optimal_ratio_experiment.csv"

retrieve(directory, version, output_dir, output_fname)

# After retrieving, run the bash command
# rsync -zarv --prune-empty-dirs --include="*/" --include="*.csv" --exclude="*" /scratch/public/jitter/wt/experiment_results/ /work/yl708/experiment_outputs
# to copy over all the csv files
# Source : https://stackoverflow.com/questions/11111562/rsync-copy-over-only-certain-types-of-files-using-include-option

