import glob
import os
import retrieve_maps

#directory holding experimental outputs
directory = "/scratch/public/MW_batch_size_8_test/"

#Will likely stay some- specified in run save train test py
version = "v2"

#Directory and csv file name to output to
output_dir = "/home/fcw/scripts/experiment_csvs/"
output_fname = "mw_bs_8.csv"

retrieve(directory, version, output_dir, output_fname)