import fileinput
import sys


path = '/hdd/dataplus2021/whtest/repro_bass_300/domain_experiment/BC_team_domain_experiment/'

files = ['Train EM Val EM 100 real 75 syn/baseline/val_img_paths.txt', 'Train EM Val EM 100 real 75 syn/baseline/val_lbl_paths.txt',
        'Train EM Val NE 100 real 75 syn/baseline/val_img_paths.txt', 'Train EM Val NE 100 real 75 syn/baseline/val_lbl_paths.txt',
        'Train NE Val EM 100 real 75 syn/baseline/val_img_paths.txt', 'Train NE Val EM 100 real 75 syn/baseline/val_lbl_paths.txt',
        'Train NE Val NE 100 real 75 syn/baseline/val_img_paths.txt', 'Train NE Val NE 100 real 75 syn/baseline/val_lbl_paths.txt']

for file in files:
    for line in fileinput.input(path+file, inplace=1):
        if 'ak478' in line:
            line = line.replace('ak478', 'whtest')
        sys.stdout.write(line)


# import glob

# for file in glob.iglob(r'/hdd/dataplus2021/whtest/repro_bass_300/domain_txt_files/*.txt'):
#     for line in fileinput.input(file, inplace=1):
#         if '/share/data/' in line:
#             line = line.replace('/share/data/', '/share/domain_experiment/BC_team_domain_experiment/data/')
#         sys.stdout.write(line)