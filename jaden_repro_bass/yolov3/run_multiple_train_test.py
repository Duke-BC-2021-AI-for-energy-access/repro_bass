import os
import subprocess


class Dataset():
    def __init__(self, img_txt, lbl_txt, out_dir, img_txt_val, lbl_txt_val):
        self.img_txt = img_txt
        self.lbl_txt = lbl_txt
        self.out_dir = out_dir
        self.img_txt_val = img_txt_val
        self.lbl_txt_val = lbl_txt_val

    def get_img_txt(self):
        return self.img_txt
    
    def get_lbl_txt(self):
        return self.lbl_txt
    
    def get_out_dir(self):
        return self.out_dir
    
    def get_img_txt_val(self):
        return self.img_txt_val
    
    def get_lbl_txt_val(self):
        return self.lbl_txt_val

# SWITCH
#out_path = '/hdd/dataplus2021/share/cross_val/'
#train_path = '/hdd/dataplus2021/whtest/repro_bass_300/domain_txt_files/'
#val_path = '/hdd/dataplus2021/whtest/repro_bass_300/domain_experiment/BC_team_domain_experiment/'

#out_path = '/hdd/dataplus2021/share/cross_val/'
#train_path = '/hdd/dataplus2021/ak478/repro_bass_300/domain_txt_files2/'
#val_path = '/hdd/dataplus2021/ak478/repro_bass_300/domain_experiment/BC_team_domain_experiment/'

out_path = '/scratch/public/dataplus_batch_8/'
train_path = '/home/fcw/updated_txt_files_2/'
val_path = '/home/fcw/repro_bass_300/domain_experiment/BC_team_domain_experiment/'

# OG DOMAINS
t_EM_v_EM_0 = Dataset(img_txt=train_path+'train_EM_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_EM_lbls.txt',
                    out_dir=out_path+'t_EM_v_EM_0/',
                    img_txt_val=val_path+'Train EM Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_EM_1 = Dataset(img_txt=train_path+'train_EM_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_EM_lbls.txt',
                    out_dir=out_path+'t_EM_v_EM_1/',
                    img_txt_val=val_path+'Train EM Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_EM_2 = Dataset(img_txt=train_path+'train_EM_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_EM_lbls.txt',
                    out_dir=out_path+'t_EM_v_EM_2/',
                    img_txt_val=val_path+'Train EM Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_EM_3 = Dataset(img_txt=train_path+'train_EM_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_EM_lbls.txt',
                    out_dir=out_path+'t_EM_v_EM_3/',
                    img_txt_val=val_path+'Train EM Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')

t_EM_v_NE_0 = Dataset(img_txt=train_path+'train_EM_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_NE_lbls.txt',
                    out_dir=out_path+'t_EM_v_NE_0/',
                    img_txt_val=val_path+'Train EM Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_NE_1 = Dataset(img_txt=train_path+'train_EM_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_NE_lbls.txt',
                    out_dir=out_path+'t_EM_v_NE_1/',
                    img_txt_val=val_path+'Train EM Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_NE_2 = Dataset(img_txt=train_path+'train_EM_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_NE_lbls.txt',
                    out_dir=out_path+'t_EM_v_NE_2/',
                    img_txt_val=val_path+'Train EM Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_NE_3 = Dataset(img_txt=train_path+'train_EM_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_NE_lbls.txt',
                    out_dir=out_path+'t_EM_v_NE_3/',
                    img_txt_val=val_path+'Train EM Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')


t_NE_v_EM_0 = Dataset(img_txt=train_path+'train_NE_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_EM_lbls.txt',
                    out_dir=out_path+'t_NE_v_EM_0/',
                    img_txt_val=val_path+'Train NE Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_EM_1 = Dataset(img_txt=train_path+'train_NE_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_EM_lbls.txt',
                    out_dir=out_path+'t_NE_v_EM_1/',
                    img_txt_val=val_path+'Train NE Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_EM_2 = Dataset(img_txt=train_path+'train_NE_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_EM_lbls.txt',
                    out_dir=out_path+'t_NE_v_EM_2/',
                    img_txt_val=val_path+'Train NE Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_EM_3 = Dataset(img_txt=train_path+'train_NE_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_EM_lbls.txt',
                    out_dir=out_path+'t_NE_v_EM_3/',
                    img_txt_val=val_path+'Train NE Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')

t_NE_v_NE_0 = Dataset(img_txt=train_path+'train_NE_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_NE_lbls.txt',
                    out_dir=out_path+'t_NE_v_NE_0/',
                    img_txt_val=val_path+'Train NE Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_NE_1 = Dataset(img_txt=train_path+'train_NE_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_NE_lbls.txt',
                    out_dir=out_path+'t_NE_v_NE_1/',
                    img_txt_val=val_path+'Train NE Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_NE_2 = Dataset(img_txt=train_path+'train_NE_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_NE_lbls.txt',
                    out_dir=out_path+'t_NE_v_NE_2/',
                    img_txt_val=val_path+'Train NE Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_NE_3 = Dataset(img_txt=train_path+'train_NE_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_NE_lbls.txt',
                    out_dir=out_path+'t_NE_v_NE_3/',
                    img_txt_val=val_path+'Train NE Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')


# MORE DOMAINS
t_NW_v_NW_0 = Dataset(img_txt=train_path+'train_NW_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_NW_lbls.txt',
                    out_dir=out_path+'t_NW_v_NW_0/',
                    img_txt_val=val_path+'Train NW Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_NW_1 = Dataset(img_txt=train_path+'train_NW_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_NW_lbls.txt',
                    out_dir=out_path+'t_NW_v_NW_1/',
                    img_txt_val=val_path+'Train NW Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_NW_2 = Dataset(img_txt=train_path+'train_NW_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_NW_lbls.txt',
                    out_dir=out_path+'t_NW_v_NW_2/',
                    img_txt_val=val_path+'Train NW Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_NW_3 = Dataset(img_txt=train_path+'train_NW_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_NW_lbls.txt',
                    out_dir=out_path+'t_NW_v_NW_3/',
                    img_txt_val=val_path+'Train NW Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')

t_NW_v_EM_0 = Dataset(img_txt=train_path+'train_NW_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_EM_lbls.txt',
                    out_dir=out_path+'t_NW_v_EM_0/',
                    img_txt_val=val_path+'Train NW Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_EM_1 = Dataset(img_txt=train_path+'train_NW_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_EM_lbls.txt',
                    out_dir=out_path+'t_NW_v_EM_1/',
                    img_txt_val=val_path+'Train NW Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_EM_2 = Dataset(img_txt=train_path+'train_NW_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_EM_lbls.txt',
                    out_dir=out_path+'t_NW_v_EM_2/',
                    img_txt_val=val_path+'Train NW Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_EM_3 = Dataset(img_txt=train_path+'train_NW_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_EM_lbls.txt',
                    out_dir=out_path+'t_NW_v_EM_3/',
                    img_txt_val=val_path+'Train NW Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')

t_NW_v_NE_0 = Dataset(img_txt=train_path+'train_NW_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_NE_lbls.txt',
                    out_dir=out_path+'t_NW_v_NE_0/',
                    img_txt_val=val_path+'Train NW Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_NE_1 = Dataset(img_txt=train_path+'train_NW_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_NE_lbls.txt',
                    out_dir=out_path+'t_NW_v_NE_1/',
                    img_txt_val=val_path+'Train NW Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_NE_2 = Dataset(img_txt=train_path+'train_NW_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_NE_lbls.txt',
                    out_dir=out_path+'t_NW_v_NE_2/',
                    img_txt_val=val_path+'Train NW Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_NE_3 = Dataset(img_txt=train_path+'train_NW_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_NE_lbls.txt',
                    out_dir=out_path+'t_NW_v_NE_3/',
                    img_txt_val=val_path+'Train NW Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')

t_NW_v_SW_0 = Dataset(img_txt=train_path+'train_NW_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_SW_lbls.txt',
                    out_dir=out_path+'t_NW_v_SW_0/',
                    img_txt_val=val_path+'Train NW Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_SW_1 = Dataset(img_txt=train_path+'train_NW_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_SW_lbls.txt',
                    out_dir=out_path+'t_NW_v_SW_1/',
                    img_txt_val=val_path+'Train NW Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_SW_2 = Dataset(img_txt=train_path+'train_NW_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_SW_lbls.txt',
                    out_dir=out_path+'t_NW_v_SW_2/',
                    img_txt_val=val_path+'Train NW Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NW_v_SW_3 = Dataset(img_txt=train_path+'train_NW_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_NW_val_SW_lbls.txt',
                    out_dir=out_path+'t_NW_v_SW_3/',
                    img_txt_val=val_path+'Train NW Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NW Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')


t_SW_v_SW_0 = Dataset(img_txt=train_path+'train_SW_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_SW_lbls.txt',
                    out_dir=out_path+'t_SW_v_SW_0/',
                    img_txt_val=val_path+'Train SW Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_SW_1 = Dataset(img_txt=train_path+'train_SW_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_SW_lbls.txt',
                    out_dir=out_path+'t_SW_v_SW_1/',
                    img_txt_val=val_path+'Train SW Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_SW_2 = Dataset(img_txt=train_path+'train_SW_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_SW_lbls.txt',
                    out_dir=out_path+'t_SW_v_SW_2/',
                    img_txt_val=val_path+'Train SW Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_SW_3 = Dataset(img_txt=train_path+'train_SW_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_SW_lbls.txt',
                    out_dir=out_path+'t_SW_v_SW_3/',
                    img_txt_val=val_path+'Train SW Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')

t_SW_v_EM_0 = Dataset(img_txt=train_path+'train_SW_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_EM_lbls.txt',
                    out_dir=out_path+'t_SW_v_EM_0/',
                    img_txt_val=val_path+'Train SW Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_EM_1 = Dataset(img_txt=train_path+'train_SW_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_EM_lbls.txt',
                    out_dir=out_path+'t_SW_v_EM_1/',
                    img_txt_val=val_path+'Train SW Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_EM_2 = Dataset(img_txt=train_path+'train_SW_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_EM_lbls.txt',
                    out_dir=out_path+'t_SW_v_EM_2/',
                    img_txt_val=val_path+'Train SW Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_EM_3 = Dataset(img_txt=train_path+'train_SW_val_EM_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_EM_lbls.txt',
                    out_dir=out_path+'t_SW_v_EM_3/',
                    img_txt_val=val_path+'Train SW Val EM 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val EM 100 real 75 syn/baseline/val_lbl_paths.txt')

t_SW_v_NE_0 = Dataset(img_txt=train_path+'train_SW_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_NE_lbls.txt',
                    out_dir=out_path+'t_SW_v_NE_0/',
                    img_txt_val=val_path+'Train SW Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_NE_1 = Dataset(img_txt=train_path+'train_SW_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_NE_lbls.txt',
                    out_dir=out_path+'t_SW_v_NE_1/',
                    img_txt_val=val_path+'Train SW Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_NE_2 = Dataset(img_txt=train_path+'train_SW_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_NE_lbls.txt',
                    out_dir=out_path+'t_SW_v_NE_2/',
                    img_txt_val=val_path+'Train SW Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_NE_3 = Dataset(img_txt=train_path+'train_SW_val_NE_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_NE_lbls.txt',
                    out_dir=out_path+'t_SW_v_NE_3/',
                    img_txt_val=val_path+'Train SW Val NE 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val NE 100 real 75 syn/baseline/val_lbl_paths.txt')

t_SW_v_NW_0 = Dataset(img_txt=train_path+'train_SW_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_NW_lbls.txt',
                    out_dir=out_path+'t_SW_v_NW_0/',
                    img_txt_val=val_path+'Train SW Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_NW_1 = Dataset(img_txt=train_path+'train_SW_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_NW_lbls.txt',
                    out_dir=out_path+'t_SW_v_NW_1/',
                    img_txt_val=val_path+'Train SW Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_NW_2 = Dataset(img_txt=train_path+'train_SW_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_NW_lbls.txt',
                    out_dir=out_path+'t_SW_v_NW_2/',
                    img_txt_val=val_path+'Train SW Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_SW_v_NW_3 = Dataset(img_txt=train_path+'train_SW_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_SW_val_NW_lbls.txt',
                    out_dir=out_path+'t_SW_v_NW_3/',
                    img_txt_val=val_path+'Train SW Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train SW Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')


t_EM_v_NW_0 = Dataset(img_txt=train_path+'train_EM_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_NW_lbls.txt',
                    out_dir=out_path+'t_EM_v_NW_0/',
                    img_txt_val=val_path+'Train EM Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_NW_1 = Dataset(img_txt=train_path+'train_EM_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_NW_lbls.txt',
                    out_dir=out_path+'t_EM_v_NW_1/',
                    img_txt_val=val_path+'Train EM Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_NW_2 = Dataset(img_txt=train_path+'train_EM_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_NW_lbls.txt',
                    out_dir=out_path+'t_EM_v_NW_2/',
                    img_txt_val=val_path+'Train EM Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_NW_3 = Dataset(img_txt=train_path+'train_EM_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_NW_lbls.txt',
                    out_dir=out_path+'t_EM_v_NW_3/',
                    img_txt_val=val_path+'Train EM Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')

t_EM_v_SW_0 = Dataset(img_txt=train_path+'train_EM_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_SW_lbls.txt',
                    out_dir=out_path+'t_EM_v_SW_0/',
                    img_txt_val=val_path+'Train EM Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_SW_1 = Dataset(img_txt=train_path+'train_EM_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_SW_lbls.txt',
                    out_dir=out_path+'t_EM_v_SW_1/',
                    img_txt_val=val_path+'Train EM Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_SW_2 = Dataset(img_txt=train_path+'train_EM_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_SW_lbls.txt',
                    out_dir=out_path+'t_EM_v_SW_2/',
                    img_txt_val=val_path+'Train EM Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_EM_v_SW_3 = Dataset(img_txt=train_path+'train_EM_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_EM_val_SW_lbls.txt',
                    out_dir=out_path+'t_EM_v_SW_3/',
                    img_txt_val=val_path+'Train EM Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train EM Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')

t_NE_v_NW_0 = Dataset(img_txt=train_path+'train_NE_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_NW_lbls.txt',
                    out_dir=out_path+'t_NE_v_NW_0/',
                    img_txt_val=val_path+'Train NE Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_NW_1 = Dataset(img_txt=train_path+'train_NE_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_NW_lbls.txt',
                    out_dir=out_path+'t_NE_v_NW_1/',
                    img_txt_val=val_path+'Train NE Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_NW_2 = Dataset(img_txt=train_path+'train_NE_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_NW_lbls.txt',
                    out_dir=out_path+'t_NE_v_NW_2/',
                    img_txt_val=val_path+'Train NE Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_NW_3 = Dataset(img_txt=train_path+'train_NE_val_NW_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_NW_lbls.txt',
                    out_dir=out_path+'t_NE_v_NW_3/',
                    img_txt_val=val_path+'Train NE Val NW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val NW 100 real 75 syn/baseline/val_lbl_paths.txt')

t_NE_v_SW_0 = Dataset(img_txt=train_path+'train_NE_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_SW_lbls.txt',
                    out_dir=out_path+'t_NE_v_SW_0/',
                    img_txt_val=val_path+'Train NE Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_SW_1 = Dataset(img_txt=train_path+'train_NE_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_SW_lbls.txt',
                    out_dir=out_path+'t_NE_v_SW_1/',
                    img_txt_val=val_path+'Train NE Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_SW_2 = Dataset(img_txt=train_path+'train_NE_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_SW_lbls.txt',
                    out_dir=out_path+'t_NE_v_SW_2/',
                    img_txt_val=val_path+'Train NE Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')
t_NE_v_SW_3 = Dataset(img_txt=train_path+'train_NE_val_SW_imgs.txt',
                    lbl_txt=train_path+'train_NE_val_SW_lbls.txt',
                    out_dir=out_path+'t_NE_v_SW_3/',
                    img_txt_val=val_path+'Train NE Val SW 100 real 75 syn/baseline/val_img_paths.txt',
                    lbl_txt_val=val_path+'Train NE Val SW 100 real 75 syn/baseline/val_lbl_paths.txt')


# trials = [
#         t_NW_v_NW_0, t_NW_v_NW_1, t_NW_v_NW_2, t_NW_v_NW_3, 
#         t_NW_v_EM_0, t_NW_v_EM_1, t_NW_v_EM_2, t_NW_v_EM_3, 
#         t_NW_v_NE_0, t_NW_v_NE_1, t_NW_v_NE_2, t_NW_v_NE_3,
#         t_NW_v_SW_0, t_NW_v_SW_1, t_NW_v_SW_2, t_NW_v_SW_3,
#         ]

# SWITCH
'''
trials = [
        t_SW_v_SW_0, t_SW_v_SW_1, t_SW_v_SW_2, t_SW_v_SW_3,
        t_SW_v_EM_0, t_SW_v_EM_1, t_SW_v_EM_2, t_SW_v_EM_3, 
        t_SW_v_NE_0, t_SW_v_NE_1, t_SW_v_NE_2, t_SW_v_NE_3,
        t_SW_v_NW_0, t_SW_v_NW_1, t_SW_v_NW_2, t_SW_v_NW_3,

        t_EM_v_NW_0, t_EM_v_NW_1, t_EM_v_NW_2, t_EM_v_NW_3,
        t_EM_v_SW_0, t_EM_v_SW_1, t_EM_v_SW_2, t_EM_v_SW_3, 

        t_NE_v_NW_0, t_NE_v_NW_1, t_NE_v_NW_2, t_NE_v_NW_3,
        t_NE_v_SW_0, t_NE_v_SW_1, t_NE_v_SW_2, t_NE_v_SW_3]
'''

#trials = [
#        t_EM_v_EM_0, t_EM_v_EM_1, t_EM_v_EM_2, t_EM_v_EM_3,
#        t_EM_v_NE_0, t_EM_v_NE_1, t_EM_v_NE_2, t_EM_v_NE_3]

trials = [t_EM_v_NW_0,t_EM_v_NW_1,t_EM_v_NW_2,t_EM_v_NW_3,
    t_EM_v_SW_0,t_EM_v_SW_1,t_EM_v_SW_2,t_EM_v_SW_3,
    t_NE_v_NE_0,t_NE_v_NE_1,t_NE_v_NE_2,t_NE_v_NE_3,
    t_NE_v_NW_0,t_NE_v_NW_1,t_NE_v_NW_2,t_NE_v_NW_3,
    t_NE_v_SW_0,t_NE_v_SW_1,t_NE_v_SW_2,t_NE_v_SW_3,
    t_NW_v_EM_0,t_NW_v_EM_1,t_NW_v_EM_2,t_NW_v_EM_3,
    t_NW_v_NE_0,t_NW_v_NE_1,t_NW_v_NE_2,t_NW_v_NE_3,
    t_NW_v_NW_0,t_NW_v_NW_1,t_NW_v_NW_2,t_NW_v_NW_3,
    t_NW_v_SW_0,t_NW_v_SW_1,t_NW_v_SW_2,t_NW_v_SW_3,
    t_SW_v_EM_0,t_SW_v_EM_1,t_SW_v_EM_2,t_SW_v_EM_3,
    t_SW_v_NE_0,t_SW_v_NE_1,t_SW_v_NE_2,t_SW_v_NE_3,
    t_SW_v_NW_0,t_SW_v_NW_1,t_SW_v_NW_2,t_SW_v_NW_3,
    t_SW_v_SW_0,t_SW_v_SW_1,t_SW_v_SW_2,t_SW_v_SW_3]



for trial in trials:
    subprocess.run(['python', 'run_save_train_test.py',
                    '--img_list', trial.get_img_txt(), 
                    '--lbl_list', trial.get_lbl_txt(),
                    '--epochs', '300',
                    '--out_dir', trial.get_out_dir(),
                    '--img_list_val', trial.get_img_txt_val(), 
                    '--lbl_list_val', trial.get_lbl_txt_val(),
                    '--version', 'v2'])


""" baseline_0 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_base.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_base.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/baseline_300_0/')
baseline_1 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_base.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_base.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/baseline_300_1/')
baseline_2 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_base.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_base.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/baseline_300_2/')
baseline_3 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_base.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_base.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/baseline_300_3/')

gp_gan_0 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_gen_ratio_fixed.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio_fixed.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/gp_gan_300_0/')
gp_gan_1 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_gen_ratio_fixed.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio_fixed.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/gp_gan_300_1/')
gp_gan_2 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_gen_ratio_fixed.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio_fixed.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/gp_gan_300_2/')
gp_gan_3 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_gen_ratio_fixed.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio_fixed.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/gp_gan_300_3/')

gp_gan_poly_0 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_gen_ratio_poly.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio_poly.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/gp_gan_poly_300_0/')
gp_gan_poly_1 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_gen_ratio_poly.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio_poly.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/gp_gan_poly_300_1/')
gp_gan_poly_2 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_gen_ratio_poly.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio_poly.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/gp_gan_poly_300_2/')
gp_gan_poly_3 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_img_gen_ratio_poly.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_lbl_gen_ratio_poly.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/gp_gan_poly_300_3/')


city_en_0 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_cityeng_img.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_cityeng_lbl.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/cityeng_300_0/')
city_en_1 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_cityeng_img.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_cityeng_lbl.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/cityeng_300_1/')
city_en_2 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_cityeng_img.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_cityeng_lbl.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/cityeng_300_2/')
city_en_3 = dataset(img_txt='/hdd/dataplus2021/fcw/wnd_train_cityeng_img.txt', 
                lbl_txt='/hdd/dataplus2021/fcw/wnd_train_cityeng_lbl.txt', 
                out_dir= '/hdd/dataplus2021/share/exper_v1/cityeng_300_3/') """
