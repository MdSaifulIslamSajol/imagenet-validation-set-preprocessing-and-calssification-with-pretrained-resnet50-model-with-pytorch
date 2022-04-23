# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:19:37 2022

@author: msajol1
"""



""" 
As discussed here https://discuss.pytorch.org/t/filenotfounderror-couldnt-find-any-class-folder/138578/3 
the validation images need to be set in subfolders inorder to work with torchvision.datasets.ImageFolder.

1. The main files are ILSVRC2012_img_val.rar (if you only download the validation set)
     and  ILSVRC2012_devkit_t12.tar. 
2. Download them from Imagenet 2012 dataset server
3. Extract both files.
4. ILSVRC2012_validation_ground_truth.txt  is the ground truth that comes with dev kit. 
5. The class names will be selected from annotations files (comes with 165GB package),
   and subfolders will be created based on the class names
6. ILSVRC2012_img_val  is validation images, which need to put in subfolders of a folder named "val_subfolders".
7. ILSVRC2012_img_val will eventually be empty after one run(execution).
8. For every other run you might need to extract  the ILSVRC2012_devkit_t12.tar file again and 
   delete the "val_subfolders" folder created by previous run.
9. This script particularly works  only for windows os!

""" 
import os
import time
import shutil
import pandas as pd
import xml
import os,glob,time
import shutil 

# for windows desktop 
path= r"C:\Users\msajol1\imagenet"  # root imagenet data folder
img_path = r"\Users\msajol1\imagenet\ILSVRC2012_img_val"  # ILSVRC2012_img_val folder path
val_label_path= r"\Users\msajol1\imagenet\ILSVRC2012_devkit_t12\data"  # ILSVRC2012_devkit_t12/data folder path
annotations_path= r"C:\Users\msajol1\imagenet\Annotations\val_ant\val"  # /ILSVRC/Annotations/CLS-LOC/val/"
csv_file_saving_path= path

# # for server (doesn't work this code properly on linux)
# root_path=os.getcwd()
# path= "/data/imagenet_datasets/"   # root imagenet data folder
# img_path = "/data/imagenet_datasets/ILSVRC2012_img_val/"   # ILSVRC2012_img_val folder path
# val_label_path= "/data/imagenet_datasets/ILSVRC2012_devkit_t12/data/"   # ILSVRC2012_devkit_t12/data folder path
# annotations_path= "/data/imagenet_datasets/ILSVRC/Annotations/CLS-LOC/val/"  # /ILSVRC/Annotations/CLS-LOC/val/"
# csv_file_saving_path= path


# =============================================================================
#  extracting class name from annotation folder
# =============================================================================

files_in_folder = os.listdir(annotations_path)

img_nm_list=[]
clss_list=[]
data2 = pd.DataFrame(columns = ['img_id', 'class_nm']) 

for file in glob.glob(os.path.join(annotations_path, "*.xml")):
    df = pd.read_xml(file)  
    img_nm=df['filename'].iloc[1]
    clss=df['name'].iloc[5]
    img_nm_list.append(img_nm)
    clss_list.append(clss)

df1 = pd.DataFrame (img_nm_list, columns = ['img_id'])
df2 = pd.DataFrame (clss_list, columns = ['class_nm'])
df=pd.concat([df1,df2],axis=1, join='inner')
csv_path=os.path.join(path, "validation_img_name_and_class.csv")
df.to_csv(csv_path,header=False)  #ILSVRC2012_val_00000001 , n01751748

# =============================================================================
#  creating a dictionary of img names and their corresponding class
# =============================================================================

# Check whether "val_subfolders" folder already exists or not, if exists remove the old one
# dirpath = os.path.join(path, "val_subfolders")
# if os.path.exists(dirpath) and os.path.isdir(dirpath):
#     print("val already Exist")
#     shutil.rmtree(dirpath)
#     print("old val folder removed")
    
image_name_list = os.listdir(img_path)  # converting images in to list
print(image_name_list[0:5])
val_subf_img_dir = os.path.join(path, "val_subfolders")

# Open and read val label 
fp = open(os.path.join(val_label_path, 'ILSVRC2012_validation_ground_truth.txt'), 'r')
label_list = fp.readlines()

print("label_list[0:5]" ,label_list[0:5])
print("clss_list[0:5]", clss_list[0:5])

# Create dictionary to store img filename and corresponding
val_img_dict = {}
for idx, label in enumerate(clss_list):
    key=image_name_list[idx]
    val_img_dict[key] = label
fp.close()

# Display first 10 entries of resulting val_img_dict dictionary
print("# val_img_dict (first five pairs)")
print({k: val_img_dict[k] for k in list(val_img_dict)[:5]})

# =============================================================================
#   creating subfolders
# =============================================================================
#Create subfolders (if not present) for validation images based on label,
#    and move images into the respective folder    
for img, folder in val_img_dict.items():
    subfolder = (os.path.join(val_subf_img_dir, folder))
    
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
        
    if os.path.exists(os.path.join(img_path, img)) :  # if the image remains in ILSVRC2012_img_val

        # os.rename(os.path.join(img_path, img), #old directory
        #           os.path.join(subfolder, img))  # new directory

        source=os.path.join(img_path, img)
        destination=os.path.join(subfolder, img)
        shutil.copy(source, destination)
        
