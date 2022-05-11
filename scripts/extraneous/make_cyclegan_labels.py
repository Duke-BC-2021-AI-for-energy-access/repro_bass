#import glob
import os
import glob
from shutil import copyfile
import re

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

reg_dict = {
        ".jpg": ".txt",
        "/images": "/labels"
}

cycle_dir = "/scratch/cek28/cyclegan-output/images/"

labels_dir = "/scratch/cek28/domain_experiment/BC_team_domain_experiment/data/labels/"

all_labels = glob.glob(labels_dir+ "*.txt")

for root, subdirectories, files in os.walk(cycle_dir):
    for subdirectory in subdirectories:
        output_dir = os.path.join(root, subdirectory).replace("/images", "/labels") + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

cycle_files = glob.glob(cycle_dir + "**/*.jpg", recursive=True)
  
for file in cycle_files:
    fname = file[file.rfind("/")+1:]
    label_fname = fname.replace(".jpg", ".txt")
    true_label = labels_dir + label_fname

    output_label = multiple_replace(reg_dict, file)
    #print(true_label, output_label)
    copyfile(true_label, output_label)
