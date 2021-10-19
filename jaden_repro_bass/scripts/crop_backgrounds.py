import glob
from PIL import Image
import os

img_directory = "/scratch/public/MW_images/train_background/"
all_imgs = glob.glob(img_directory + "*.jpg")

results_dir ="/scratch/public/MW_images/train_background_resized/"

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

for img_path in all_imgs:
    img = Image.open(img_path)
    cropped_img = img.resize((608,608))
    fname = img_path[img_path.rfind("/")+1:]
    cropped_img_path = results_dir + fname
    print(cropped_img_path)
    cropped_img.save(cropped_img_path)