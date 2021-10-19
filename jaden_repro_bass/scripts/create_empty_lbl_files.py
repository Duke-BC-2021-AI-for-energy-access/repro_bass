reg_file = "/home/fcw/scripts/nes/reg_nes.txt"

with open(reg_file, 'r') as f:
    for file in f.read().split("\n"):
        with open(file, 'w') as fp:
            pass

val_file = "/home/fcw/scripts/nes/val_nes.txt"

with open(val_file, 'r') as f:
    for file in f.read().split("\n"):
        with open(file, 'w') as fp:
            pass