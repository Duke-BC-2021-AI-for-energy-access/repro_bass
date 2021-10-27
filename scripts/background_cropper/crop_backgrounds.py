import glob
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description='Cropper for background images to 608x608')
parser.add_argument('--crop_flag', type=str, default='middle', help='new domain to add')
parser.add_argument('--new_width', type=int, default=608, help='new domain to add')
parser.add_argument('--new_height', type=int, default=608, help='new domain to add')
parser.add_argument('--input_dir', type=str, default="/scratch/public/MW_images/0m_train/", help='new domain to add')
parser.add_argument('--output_dir', type=str, default="/scratch/public/MW_images/0m_train_cropped/", help='new domain to add')

args = parser.parse_args()
crop_flag = args.crop_flag
new_width = args.new_width
new_height = args.new_height

#img_directory = "/scratch/public/MW_images/train_background/"
img_directory = args.input_dir
results_dir = args.output_dir

all_imgs = glob.glob(img_directory + "*.jpg")

#results_dir ="/scratch/public/MW_images/train_background_cropped/"


if not os.path.exists(results_dir):
    os.mkdir(results_dir)

for img_path in all_imgs:
    img = Image.open(img_path)
    #cropped_img = img.resize((608,608))

    #Top left background
    #cropped_img = img.crop((0, 0, 608, 608))
    width, height = img.size

    if crop_flag == "middle":
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
    else:
        left, top, right, bottom = 0

    cropped_img = img.crop((left, top, right, bottom))
    fname = img_path[img_path.rfind("/")+1:]
    cropped_img_path = results_dir + fname
    print(cropped_img_path)
    cropped_img.save(cropped_img_path)