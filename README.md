# imagenet-validation-set-preprocessing-and-calssification-with-pretrained-resnet50-model-with-pytorch
This is an example of how to preprocess imagenet validation set before loading it to the dataloader.
Additionally the loaded dataset was tested with a pre-trained resnet50 model for classification
number of classes=1000
As discussed here https://discuss.pytorch.org/t/filenotfounderror-couldnt-find-any-class-folder/138578/3 
the validation images need to be set in subfolders inorder to work with torchvision.datasets.ImageFolder.
1. The main files are: 
    #### ILSVRC2012_img_val.rar ( download the validation set) 
    #### ILSVRC2012_devkit_t12.tar.
    #### ILSVRC (comes with 155GB package)
    
2. Download them from Imagenet 2012 dataset server
3. Extract all files.
4. ILSVRC2012_validation_ground_truth.txt  is the ground truth that comes with dev kit. 
5. The class names will be selected from annotations files (
    from  "ILSVRC"/"Annotations"/"CLS-LOC"/ "val" directory) ,
   and subfolders will be created based on the class names
6. ILSVRC2012_img_val  contains validation images, which need to put inside seperate  subfolders 
7. All subfolders will be put inside a parent folder named "val_subfolders".
8. For every other run you should delete the "val_subfolders" folder first created by previous run.

