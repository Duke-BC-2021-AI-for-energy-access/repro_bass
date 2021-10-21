import glob

for file in glob.iglob(r'/hdd/dataplus2021/ak478/repro_bass_300/domain_txt_files2/*.txt'):
    shape_name = file.split('/')[-1][:-3]+'shapes'
    
    try:
        f = open('/hdd/dataplus2021/ak478/repro_bass_300/domain_txt_files2/' + shape_name, 'r+')
        f.truncate(0)
    except:
        pass

    with open('/hdd/dataplus2021/ak478/repro_bass_300/domain_txt_files2/' + shape_name, 'w') as f:
        for i in range(100):
            f.write('608 608\n')
        f.write('\n')
        for i in range(75):
            f.write('608 608\n')
