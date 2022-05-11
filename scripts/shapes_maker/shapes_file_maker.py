import glob

baseline_files = glob.glob("/scratch/cek28/txt_files/25background_experiment/*.txt")

img_files = [x for x in baseline_files if "img" in x]

for img in img_files:
    with open(img, "r") as f:
        lines = f.read().split('\n')
        lines.remove('')
        line_num = len(lines)
    with open(img.replace(".txt", ".shapes"), "w") as f:
        for i in range(0, line_num-1):
            f.writelines("608 608\n")
        f.writelines("608 608")