import glob
import os
from retrieve_maps import retrieve

#directory holding experimental outputs
directory = "/scratch/public/jitter/wt/experiment_results/Optimal_Ratio_6/"

#Will likely stay some- specified in run save train test py
version = "v2"

#Directory and csv file name to output to
# output_dir = "/scratch/public/jitter/wt/experiment_results"
output_dir = directory ## outputting the results to wherever we ran the experiment
output_fname = "optimal_ratio_experiment.csv"

retrieve(directory, version, output_dir, output_fname)
