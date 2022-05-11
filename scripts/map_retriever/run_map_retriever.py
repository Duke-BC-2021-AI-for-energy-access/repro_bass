import glob
import os
from retrieve_maps import retrieve

#directory holding experimental outputs
directory = "/scratch/cek28/results/cyclegan/"

#Will likely stay some- specified in run save train test py
version = "v2"

#Directory and csv file name to output to
output_dir = "/home/fcw/scripts/experiment_csvs/"
output_fname = "cyclegan_experiments.csv"

retrieve(directory, version, output_dir, output_fname)