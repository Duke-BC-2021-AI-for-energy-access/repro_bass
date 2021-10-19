import glob
from PIL import Image
import os

#img_directory = "/scratch/public/MW_images/train_background/"
img_directory = "/scratch/public/MW_images/0m_train/"
all_imgs = glob.glob(img_directory + "*.jpg")

results_dir = "/scratch/public/MW_images/0m_train_cropped/"
#results_dir ="/scratch/public/MW_images/train_background_cropped/"

new_width = 608
new_height = 608

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

for img_path in all_imgs:
    img = Image.open(img_path)
    #cropped_img = img.resize((608,608))

    #Top left background
    #cropped_img = img.crop((0, 0, 608, 608))
    width, height = img.size

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    cropped_img = img.crop((left, top, right, bottom))
    fname = img_path[img_path.rfind("/")+1:]
    cropped_img_path = results_dir + fname
    print(cropped_img_path)
    cropped_img.save(cropped_img_path)