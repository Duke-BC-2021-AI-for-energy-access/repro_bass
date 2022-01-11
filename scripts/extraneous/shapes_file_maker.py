import glob

baseline_files = glob.glob("/scratch/public/txt_files/MW_baseline/*.txt")

img_files = [x for x in baseline_files if "img" in x]

for img in img_files:
    with open(img.replace(".txt", ".shapes"), "w") as f:
        for i in range(0, 99):
            f.writelines("608 608\n")
        f.writelines("608 608")