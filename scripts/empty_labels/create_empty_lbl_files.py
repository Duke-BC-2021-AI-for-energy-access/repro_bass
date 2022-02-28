reg_file = "/home/fcw/scripts/nes/reg_nes.txt"

def create_labels(reg_file):
    """

    Given a file holding images to create empty labels for, creates empty labels for them

    Args:
        reg_file ([type]): File holding files to make empty txt files to create
    """
    with open(reg_file, 'r') as f:
        for file in f.read().split("\n"):
            with open(file, 'w') as fp:
                pass