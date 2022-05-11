import glob
from statistics import mean, stdev
import os
from matplotlib import pyplot as plt
import numpy as np

lbl_directory = "/scratch/dataplus2021/data/labels/"
output_directory = "/scratch/cek28/distributions/"

def plot_histogram(arr, title, output_directory, fig_num, bins):
    """[summary]

    Args:
        arr ([type]): [description]
        title ([type]): [description]
        output_directory ([type]): [description]
        fig_num ([type]): [description]
        bins ([type]): [description]
    """
    plt.figure(fig_num)
    fig = plt.hist(arr, bins=bins)
    plt.title('{title} Distribution in Real Images: Mean: {u} SD: {sd}'.format(title=title,u=round(mean(arr),3),sd=round(stdev(arr),3)))
    #plt.xlim(xmin=0, xmax = 10)
    plt.xlabel(title)
    plt.ylabel("Frequency")
    plt.savefig("{output_dir}{title}_distribution.png".format(output_dir=output_directory,title=title))

def run_distributions(lbl_directory, output_directory):
    """[summary]

    Args:
        lbl_directory ([type]): [description]
        output_directory ([type]): [description]
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    lbl_files = glob.glob(lbl_directory + "*.txt")

    num_images_arr = []
    width_arr = []
    height_arr = []

    for my_txt_file in lbl_files:

        with open(my_txt_file, "r") as f:
            lst = [float(x) for x in f.read().split()]
        
        out_ht = 608
        out_w = 608

        my_widths = [round(i*out_w) for i in lst[3::5]]
        my_heights = [round(i*out_ht)  for i in lst[4::5]]

        if len(my_widths) != 0:
            num_images_arr.append(len(my_widths))
            width_arr.extend(my_widths)
            height_arr.extend(my_heights)

    images_with_turbine_count = len(num_images_arr)

    plot_histogram(width_arr, "Width", output_directory, 1, 20)
    plot_histogram(height_arr, "Height", output_directory, 2, 20)
    plot_histogram(num_images_arr, "Number of Turbines", output_directory, 3, 50)

run_distributions(lbl_directory, output_directory)