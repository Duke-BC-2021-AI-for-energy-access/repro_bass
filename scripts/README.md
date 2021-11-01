# The scripts folder

This is an overview of the utility scripts that were made to facilitate the process of running the experiments. *Frankie fill the following in*

## A one-line summary of what each of the scripts does:

- background_cropper- Crops background images in background directory to given height/width and outputs to new folder; can crop middle or top left of image                                             
- bounding_boxs- Plots bounding boxes for images given image and YOLO txt file directories                                            
- collect_backgrounds.py-                                          
- create_empty_lbl_files.py- Reads in files without a txt file (images with no turbine), creates a empty txt file to serve as a lbl for each image                                      
- get_files_annotate.py- Given a txt file holding the images you are to annotate, it copies the files to a separate directory (so they can be downloaded all at once in a directory)                                           
- map_retriever- Given an experimental output directory, it creates a CSV holding source domain, target domain, trial number, and average precision for each trial (for easier data entry into main excel)                                                   
- mask_tester- Given a directory holding images/masks, it uses the bitwise and command to output and show what the image looks like in the context of the mask (the parts of the image corresponding to the white sections of the mask)                                                    
- old_scripts                                                    
- README.md                                                      
- repro_bass_updater- Used to update file names in directories and scripts before version control was working                                              
- txt_file_generator- Generalizable script meant for generating YOLO txt files for both training and validation for an experimental setup containing various domain combinations with baseline/supplementary imagery
