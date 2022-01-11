import glob
import cv2

my_dir = "/hdd/dataplus2021/fcw/ImageAugment4/results9/"
res_dir = "/hdd/dataplus2021/fcw/MaskTester/results/"


def test_all_masks(my_dir, res_dir):
    img_files = glob.glob(my_dir + "*.jpg")
    mask_files = [x.replace(".jpg", "__mask2.png") for x in img_files]

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    for i in range(len(img_files)):
        current_img = img_files[i]
        img = cv2.imread(current_img)
        mask = cv2.imread(mask_files[i],0)
        res = cv2.bitwise_and(img,img,mask = mask)
        img_addr = current_img[current_img.rfind("/")+1:]
        #print("/hdd/dataplus2021/fcw/MaskTester/results" + img_addr)
        cv2.imwrite(res_dir + img_addr, res)