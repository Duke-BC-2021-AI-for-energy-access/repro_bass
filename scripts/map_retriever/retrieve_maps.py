import glob
import os

def retrieve(directory, version, output_dir, output_fname):
    """
    Retrieves mAP information from a directory holding many YOLO trial outputs
    Extracts from file test_results.txt, can be adapted to extract mP, mA, mF1 

    Args:
        directory ([type]): YOLO results holding various trials (to find stats)
        version ([type]): Version specified in run save train test py
        output_dir ([type]): Directory to output csv to
        output_fname ([type]): Filename for output csv
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(output_dir + " directory was made")

    my_files = glob.glob(directory + "**/v2_outputs/test_results.txt", recursive=True)
    my_files = sorted(my_files)

    #print(my_files)

    csv_headers = "Train, Val, Trial, AP\n"

    #CREATE CSV
    out_f = open(output_dir + output_fname, "w")
    out_f.write(csv_headers)

    #Assumes version = v2
    for file in my_files:
        #Extract src, destination, trial
        print(file)
        trial_string = file[len(directory):file.find("/"+ version +"_outputs")]
        trial_parts = trial_string.split("_")
        train = trial_parts[1]
        val = trial_parts[3]
        trial = trial_parts[4]

        #READ IN 3rd value of txt file
        with open(file, "r") as f:
            mAP = f.read().split("\n")[2]
            print(mAP)
        
        out_string = "{train},{val},{trial},{map}\n".format(train=train,val=val,trial=trial,map=mAP)
        out_f.write(out_string)

    out_f.close()