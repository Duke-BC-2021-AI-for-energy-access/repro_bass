import itertools

img_file = "/scratch/cek28/txt_files/MW_baseline/MW_baseline_imgs.txt"
lbl_file = "/scratch/cek28/txt_files/MW_baseline/MW_baseline_lbls.txt"

with open(img_file, "r") as f:
    imgs = f.read()

with open(lbl_file, "r") as f:
    lbls = f.read()

domains = ["EM", "NE", "NW", "SW", "MW"]

combinations = list(itertools.product(domains, repeat=2))

def containsMW(element):
    return "MW" in element[0]

#Gets combinations with MW in it
MW_combos = list(filter(containsMW, combinations))

for combo in MW_combos:
    with open("/scratch/cek28/txt_files/MW_baseline/train_{src}_val_{dst}_imgs.txt".format(src= combo[0], dst=combo[1]), "w") as f:
        f.write(imgs)
    with open("/scratch/cek28/txt_files/MW_baseline/train_{src}_val_{dst}_lbls.txt".format(src= combo[0], dst=combo[1]), "w") as f:
        f.write(lbls)