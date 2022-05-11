import glob
from PIL import Image
import os
import subprocess

#CROP_FLAG- either 'middle' or 'left'
#If 'left',then top left corner

subprocess.run(['python', 'crop_backgrounds.py',
                '--crop_flag', 'middle', 
                '--new_width', 608,
                '--new_height', 608, 
                '--input_dir', "/scratch/cek28/MW_images/0m_train/",
                '--output_dir', "/scratch/cek28/MW_images/0m_train/"])