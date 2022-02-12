from os.path import exists
from shutil import copyfile
import os

def get_files(files, train, results):
    """

    Given an array of file names to annotate, it will grab the same files from a train directory
    and output them to a results directory

    Args:
        files ([type]): Array of file names
        train ([type]): Directory holding training images for SPECIFIC domain
        results ([type]): Result directory to output images to
    """
    if not os.path.exists(results):
        os.mkdir(results)

    for file in files:
        if exists(train + file):
            copyfile(train + file, results + file)
        #elif exists(test_dir + file):
        #    copyfile(test_dir + file, results_dir + file)