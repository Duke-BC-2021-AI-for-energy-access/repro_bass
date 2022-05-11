import random
import itertools
import os
import glob
import argparse
from shutil import copyfile

####FUNCTIONS
def containsNewDomain(domain, element):
    return domain in element

def readDomains(domains, txt_files_dir, directory=None):
    """

    Outputs dictionary for real images, real labels, supplementary images, supplementary labels
        where key is domain, value holds a list of the images/labels

    Args:
        domains ([type]): List of domains to read through
        txt_files_dir ([type]): Directory holding label files for within domain exeriments
        directory ([type], optional): Directory holding supplementary (often synthetic) images, 
            if None will sample 75 image for it

    Returns:
        [type]: [description]
    """

    real_imgs = dict()
    real_lbls = dict()
    synth_imgs = dict()
    synth_lbls = dict()

    supplementary_flag = directory is not None

    #Extracts real and synthetic images from txt files
    for domain in domains:
        within_img_file = txt_files_dir + "train_{domain}_val_{domain}_imgs.txt".format(domain=domain)
        with open(within_img_file, "r") as f:
            file_txt = list(filter(None, f.read().split("\n")))
            real_imgs[domain] = file_txt[:100]
            real_lbls[domain]  = [x.replace(".jpg", ".txt") for x in real_imgs[domain]]
            if not supplementary_flag:
                synth_imgs[domain] = file_txt[100:]
                #assumes correct naming convention
                synth_lbls[domain]  = [x.replace(".jpg", ".txt") for x in synth_imgs[domain]]
            else:
                sampleSupplementary(directory, 75, domain)
    return real_imgs, real_lbls, synth_imgs, synth_lbls

def readVals(domains, val_dir):
    """

    Unnecessary due to readDomains
    Outputs dictionary for val images, val labels
        where key is domain, value holds a list of the images/labels


    Args:
        domains ([type]):  List of domains to read through, get val for 
        val_dir ([type]): Directory holding val images

    Returns:
        [type]: [description]
    """
    val_imgs = dict()
    val_lbls = dict()
    img_fname = "val_img_paths.txt"
    lbl_fname = "val_lbl_paths.txt"
    for domain in domains:
        within_val_img_file = "{val_dir}Train {domain} Val {domain} 100 real 75 syn/baseline/{img_fname}".format(val_dir=val_dir,domain=domain,img_fnmae=img_fname)
        within_val_lbl_file = within_val_img_file.replace(img_fname, lbl_fname)
        with open(within_img_file, "r") as f:
            file_txt = list(filter(None, f.read().split("\n")))
            val_imgs[domain] = file_txt
        #could assume correct naming convention and just use generator
        with open(within_lbl_file, "r") as f:
            file_txt = list(filter(None, f.read().split("\n")))
            val_lbls[domain] = file_txt
    return val_imgs, val_lbls

def multiple_replace(dict, text):
  """
  Applies multiple replaces in string based on dictionary

  Args:
      dict ([type]): Dictionary with keys as phrase to be replaced, vals as phrase to replace key
      text ([type]): Text to apply string replaces to

  Returns:
      [type]: [description]
  """
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def sampleSupplementary(directory, n, domain,synth_imgs,synth_lbls, replacer):
    """[summary]

    Args:
        directory ([type]): Directory holding all synthetic images (assumed to hold all domains synthetic subfolders)
        n ([type]): Number of images to sample
        domain ([type]): Domain to sample supplementary for 
        synth_imgs ([type]): Dictionary holding domains as keys, list of synthetic images for domain as val
        synth_lbls ([type]): Dictionary holding domains as keys, list of synthetic labels for domain as val
        replacer ([type]): Dictionary used for string replacements for labels (images to labels, jpg to txt)
    """
    all_synth_imgs = glob.glob(directory + "/**/*.jpg", recursive = True)
    #SAMPLES WITHOUT REPLACEMENT
    synth_sample_imgs = random.sample(all_synth_imgs, n)
    synth_imgs[domain] = synth_sample_imgs
    synth_lbls[domain] = [multiple_replace(x, replacer) for x in synth_imgs[domain]]

def addDomain(domain,domains_list, domain_real_dir, domain_synth_dir, real_imgs, synth_imgs, real_lbls, synth_lbls, real_n, supp_n, val_domain):
    ####DOMAIN TO SAMPLE FROM
    #all_synth_imgs = glob.glob(domain_synth_dir + "/**/*.jpg", recursive = True)

    supplementary_replacer = {
        ".jpg": ".txt"
    }

    sampleSupplementary(domain_synth_dir+domain, supp_n, domain,synth_imgs,synth_lbls)

    #Gets 75 synthetic images
    #synth_sample_imgs = random.sample(all_synth_imgs, 75)

    #Adds MW to domains to inlcude all domains
    domains_list.append(domain)

    #Gets combinations with MW in it
    #MW_combos = list(filter(containsMW, combinations))
    #Depends on random seed

    real_replacer = {
        ".jpg": ".txt"
    }

    ###USING SAMPLE SUPPLEMENTARY TO SAMPLE 100 REAL

    #Gets 100 real images
    sampleSupplementary(domain_real_dir, real_n, val_domain, real_imgs, real_lbls)
    #domain_real_imgs = glob.glob(domain_real_dir + "*.jpg")
    #Should sample these

    #Adds MW to dictionary
    #real_imgs[domain] = domain_real_imgs
    #synth_imgs[domain] = synth_sample_imgs
    #real_lbls[domain] = [x.replace(".jpg", ".txt") for x in real_imgs[domain]]
    #synth_lbls[domain] = [x.replace(".jpg", ".txt") for x in synth_imgs[domain]]

def create_a_file(real, synth, real_files, synth_files, train_or_val, img_or_lbl, exper_dir):
    my_real_imgs = real_files[real]
    my_synth_imgs = synth_files[synth]

    #BCZ_team_domain_experiment /Train EM Val MW/folder/training_imgs.txt, training_lbls.txt, val_imgs.txt, val_lbl.txt
    #If val, don't write synthetic, or pass synthetic as none
    #if statement
    
    #pass in experiment directory (e.g. /scratch/cek28/frankie_repro_bass/domain_experiment/BC_team_domain_experiment)
    
    fname = "{train_or_val}_{img_or_lbl}_paths.txt".format(train_or_val=train_or_val, img_or_lbl=img_or_lbl)
    output_directory = "{exper_dir}/Train {real} Val {synth} 100 real 75 syn/".format(exper_dir=exper_dir,real=real,synth=synth)
    if not os.path.exists(output_directory):
        os.mkdir(results_dir)

    output_file = "{output_directory}{fname}".format(output_directory=output_directory,fname=fname)

    #/output_folder/train_EM_val_MW_imgs.txt
    #fname = "train_{real}_val_{synth}_{img_or_lbl}s.txt".format(real=real, synth=synth, img_or_lbl=img_or_lbl)

    with open(output_file, "w") as f:
        f.writelines("%s\n" % l for l in my_real_imgs)
        f.writelines("%s\n" % l for l in my_synth_imgs)