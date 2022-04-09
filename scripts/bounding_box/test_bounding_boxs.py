import cv2
import random
import glob
import os
import re

def plot_one_box(x_ctrs, y_ctrs, widths, heights, img, fname, color=None, label=None, line_thickness=None):
  """
  Plots boxes in one image

  Args:
      x_ctrs ([type]): Holds x centerpoints of YOLO bounding boxes in image
      y_ctrs ([type]): Holds y centerpoints of YOLO bounding boxes in image
      widths ([type]): Holds widths of YOLO bounding boxes in image
      heights ([type]): Holds heights of YOLO bounding boxes in image
      img ([type]): Holds image to add bounding boxes to
      fname ([type]): File name for output image
      color ([type], optional): [description]. Defaults to None.
      label ([type], optional): [description]. Defaults to None.
      line_thickness ([type], optional): [description]. Defaults to None.
  """
  for i in range(len(x_ctrs)):
    x = [x_ctrs[i]-widths[i],y_ctrs[i]-heights[i],x_ctrs[i]+widths[i],y_ctrs[i]+heights[i]]
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
  #Save file
  cv2.imwrite(fname, img)

def multiple_replace(dict, text):
  """
  Applies multiple replaces in string based on dictionary

  Args:
      dict ([type]): Dictionary with keys as phrase to be replaced, vals as phrase to replace key
      text ([type]): Text to apply string replaces to

  Returns:
      [type]: [description]
  """


  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 


#reg_dict = {
#    "ImageAugment4":"BoundingBoxes",
#    "results9":"results6"
#}

##################CHANGE
#Folder that holds images and txt files
#my_txt_dir = "/scratch/public/MW_images/test_cropped_annotated/"
#my_img_dir = "/scratch/public/MW_images/0m_test_cropped/"

##################CHANGE
#Output directory
#results_dir = "/scratch/public/scripts/bbox_test2/"

def create_boxes(txts, imgs, results_dir, dir_or_file):
  """

  1) Grabs all images and loops over them
  2) Extracts x center, y center, width, height for bounding boxes in each image
  3) Uses this information to call plot_one_box with arguments to plot bounding boxes
  for an image

  Args:
      txts ([type]): Directory holding label files for images (YOLO formatting)
      imgs ([type]): Directory holding image files or txt file holding image file names
      results_dir ([type]): Directory to output images with bounding boxes on them
  """

  if not os.path.exists(results_dir):
    os.makedirs(results_dir)

  #my_imgs = glob.glob(my_img_dir + "*.jpg")

  if not dir_or_file:
    my_txts = glob.glob(txts + "*.txt")
    my_imgs = [imgs + x[len(txts):].replace(".txt", ".jpg") for x in my_txts]
  else:
    with open(txts, "r") as f:
      my_txts = f.read().split("\n")
    with open(imgs, "r") as f:
      my_imgs = f.read().split("\n")
    
  #my_txts = [x.replace("/Cyclegan/s_EM_t_SW/", "/EM/Real/") for x in my_txts]
  #my_imgs = [x.replace("/Cyclegan/s_EM_t_SW/", "/EM/Real/") for x in my_imgs]


  #my_txts = ["/scratch/dataplus2021/data/labels/naip_2656_IA_WND_i1j1.txt", "/scratch/dataplus2021/data/labels/naip_2656_IA_WND_i0j1.txt", "/scratch/dataplus2021/data/labels/naip_1203_CA_WND_i0j0.txt"]
  #my_imgs = [x.replace("labels", "images").replace(".txt", ".jpg") for x in my_txts]

  my_imgs.sort()
  my_txts.sort()



  for k in range(len(my_imgs)):
    #Inputs- can make it glob and sort so array
    my_png_file = my_imgs[k]
    my_txt_file = my_txts[k]
    #Add output folder

    #Can loop over
    with open(my_txt_file, "r") as f:
        lst = [float(x) for x in f.read().split()]

    print(my_png_file)
    print(my_txt_file)

    my_img = cv2.imread(my_png_file)

    out_h, out_w, _ = my_img.shape

    x_ctrs = [i*out_w for i in lst[1::5]]
    y_ctrs = [i*out_h for i in lst[2::5]]
    widths = [i*(out_w / 2) for i in lst[3::5]]
    heights = [i*(out_h / 2)  for i in lst[4::5]]

    #Could add "bbox_"
    new_fname = results_dir  + my_png_file[my_png_file.rfind("/")+1:]
    #new_fname = multiple_replace(reg_dict, my_png_file)
    print(new_fname)

    plot_one_box(x_ctrs=x_ctrs, y_ctrs=y_ctrs, widths=widths, heights=heights,img=my_img,fname=new_fname)


