import argparse
import os
import subprocess
from glob import glob
import shutil 

# python run_save_train_test.py --img_list /hdd/dataplus2021/fcw/wnd_train_img_gen_ratio2.txt --lbl_list /hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio2.txt --out_dir /hdd/dataplus2021/share/exper_v0ratio2/
parser = argparse.ArgumentParser()
parser.add_argument('--img_list', type=str, default='none', help='directory of train imgs')                                             # if yes, input absolute path
parser.add_argument('--lbl_list', type=str, default='none', help='directory of train labels')                                           # if yes, input absolute path
parser.add_argument('--epochs', type=str, default='300', help='num epochs')
parser.add_argument('--out_dir', type=str, default='/home/fcw/batch_size_experiments/', help='directory to output imgs')
parser.add_argument('--img_list_val', type=str, default='none', help='directory of val imgs')                                             # if yes, input absolute path
parser.add_argument('--lbl_list_val', type=str, default='none', help='directory of val labels') 
parser.add_argument('--version', type=str, default='v0', help='version num')
parser.add_argument('--device', type=str, default='0', help='gpu id')

opt = parser.parse_args()

def make_data_file(out_root, img_list, lbl_list, version, img_list_val, lbl_list_val):
    if not os.path.exists(out_root):                                                                    # make root dir
        os.makedirs(out_root)

    try:
        f = open(out_root + 'train_data_' + version + '.data', 'r+')
        f.truncate(0)
    except:
        pass

    with open(out_root + 'train_data_' + version + '.data', 'w') as f:                                     # create master label text file
        f.write('train=' + img_list + '\n')
        f.write('train_label=' + lbl_list + '\n')
        f.write('classes=1\n')
        f.write('valid=' + img_list_val + '\n')
        f.write('valid_label=' + lbl_list_val + '\n')
        # SWITCH
        # f.write('names=/hdd/dataplus2021/whtest/repro_bass_300/yolov3/data/wnd.names\n')
        f.write('names=/home/sj305/jaden_repro_bass/yolov3/data/wnd.names\n')
        
        f.write('backup=backup/\n')
        f.write('eval=wnd')


def run_train(out_root, epochs, device):
    subprocess.run(['python', 'train.py',                                                    # train gp_gan
                    '--cfg', '/home/sj305/jaden_repro_bass/yolov3/cfg/yolov3-spp.cfg',
                    '--data', out_root + 'train_data_' + version + '.data',
                    '--img-size', '608',
                    '--epochs', epochs,
                    '--batch-size', '8',
                    '--device', device])
        


def run_test(out_root, device):
    subprocess.run(['python', 'test.py',                                                    # test gp_gan
                    '--cfg', '/home/sj305/jaden_repro_bass/yolov3/cfg/yolov3-spp.cfg',
                    '--data', out_root + 'train_data_' + version + '.data',
                    '--img-size', '608',
                    '--weights', 'weights/last.pt',
                    '--device', device])


def copy_outputs(out_root, version):
    if not os.path.exists(out_root + version + '_outputs/'):                                                                    # make root dir
        os.makedirs(out_root + version + '_outputs/')

    file_names = ['PR_curve.png', 'precision.txt', 'recall.txt', 'results.png', 'results.txt', 'test_batch0_gt.jpg', 'test_batch0_pred.jpg', 'train_batch0.jpg', 'test_results.txt', 'ious.txt']
    file_names.extend(('weights/best.pt', 'weights/last.pt'))
    for file in file_names:
        # SWITCH
        # shutil.copy2('/hdd/dataplus2021/whtest/repro_bass_300/yolov3/' + file, out_root + version + '_outputs/')

        #COPY FILE I CREATE
        shutil.copy2('/home/sj305/jaden_repro_bass/yolov3/' + file, out_root + version + '_outputs/')


img_list = opt.img_list
lbl_list = opt.lbl_list
epochs = opt.epochs
out_root = opt.out_dir
img_list_val = opt.img_list_val
lbl_list_val = opt.lbl_list_val
version = opt.version
device = opt.device

def main(img_list, lbl_list, out_root, epochs, version, device):
    make_data_file(out_root, img_list, lbl_list, version, img_list_val, lbl_list_val)
    print("Made .data file\n")
    # print((out_root, img_list, lbl_list, version, img_list_val, lbl_list_val))
    # raise Exception

    #Change back
    run_train(out_root, epochs, device)
    print("Finished Training\n")

    run_test(out_root, device)
    print("Finished Testing\n")

    copy_outputs(out_root, version)
    print("Copied outputs\n")


main(img_list, lbl_list, out_root, epochs, version, device)