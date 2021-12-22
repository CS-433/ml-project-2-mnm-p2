## DATA AUGMENTATION
"""
In this notebook we will allow for easy data augmentation implementing a function that allows for easily perform:
- Rotations: by creating more immages turning the same immage we manage to augment the flexibility of our model that becomes able to predict roads in all sort of directions.
- Mirroring: Vertical and/or horizontal pixels flipping.
- Color changing: Putting images to gray scale and/or Color Jitter (a slight change in the color values of the image)

The original 100 immages and their groud truth that are present at path_to_file, 
will be used to create a bigger training dataset. This allow the model to be less dependent on the distribution of orientation and color of the original small dataset (100 training immages is not a lot).
"""

import numpy as np
import torch 
from torchvision import transforms
import torchvision.transforms as transform
from torchvision.utils import save_image
import torch.functional as F
import PIL
import torch.utils.data as data
import os
import glob
import random 

"""We need to create an augmented set of immages. We split the data into train and 
test before augmenting the data. That way we make sure that our validation metric isn't biased by the fact that we have the same immages with small variation in the training. (eg. We have an image in the train, and the same image with a rotation of 1 degree on the test ...)
"""


def split_train_test( self,seed = 1 ,test_portion = 0.2):
    folder_path = self.folder_path

    test_set = DataLoaderSegmentation(folder_path)
    train_set = DataLoaderSegmentation(folder_path)

    copy_img_files = self.img_files
    copy_mask_files = self.mask_files

    index_array = list(range(len(copy_img_files)))
    random.Random(seed).shuffle(index_array)

    copy_img_files =  list(map(copy_img_files.__getitem__, index_array))
    copy_mask_files = list(map(copy_mask_files.__getitem__, index_array))

    quantity_test = int(len(copy_img_files)*test_portion)

    test_set.img_files = copy_img_files[:quantity_test]
    test_set.mask_files = copy_mask_files[:quantity_test]

    train_set.img_files = copy_img_files[quantity_test:]
    train_set.mask_files = copy_mask_files[quantity_test:]
    return train_set,test_set

#Inspired by https://discuss.pytorch.org/t/dataloader-for-semantic-segmentation/48290/2

#This dataloder allows to import the immages on by one, only the directories are in the RAM
class DataLoaderSegmentation(data.Dataset):

    def __init__(self, folder_path,images_folder = 'images',label_folder = 'labels',with_filter = False, filter = None):
        """Initalize the dataloader. Instead of having all the pics in ram we only save the directories."""
        super(DataLoaderSegmentation, self).__init__()
        self.folder_path = folder_path
        self.currentidx = 0
        self.img_files = glob.glob(os.path.join(folder_path,images_folder +'/','*.png'))
  
        if with_filter:
          self.img_files = [path for path in self.img_files if filter in path]
       
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,label_folder,os.path.basename(img_path))) 
       
    def __getitem__(self, index):
        """Returns only the pic,label at specified index. The returned object is a torch tensor [channels,H,W]"""
        if index >= len(self):
            index = (index%len(self))
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = transform.functional.pil_to_tensor(PIL.Image.open(img_path)).type(torch.FloatTensor)/255
        label =transform.functional.pil_to_tensor(PIL.Image.open(mask_path))[:1,:,:].type(torch.FloatTensor)/255
        return (data,label)
    def __len__(self):
        return len(self.img_files)

    def next(self,batch_size = 1):
        """Return a batch of size batch_size of pic,label from hte currentidx to currentidx + batch_size.
         The returned object is a torch tensor [batch_size,channels,H,W]"""
        if batch_size == 1:
            self.currentidx += batch_size
            return self[self.currentidx-1]
        else:
            seq = [self[self.currentidx+i] for i in range(batch_size)]
            self.currentidx += batch_size
            imgs = [i[0] for i in seq]
            lbls = [i[1] for i in seq]

            res = (torch.stack(imgs),torch.stack(lbls))
            return res

    def get_epoch(self):
        """return the current epoch"""
        return int(self.currentidx/len(self.img_files))

    def split_train_test( self,seed = 1 ,test_portion = 0.2):
        """Split dataloader into two train and test dataloaders"""
        folder_path = self.folder_path

        test_set = DataLoaderSegmentation(folder_path)
        train_set = DataLoaderSegmentation(folder_path)

        copy_img_files = self.img_files
        copy_mask_files = self.mask_files

        index_array = list(range(len(copy_img_files)))
        random.Random(seed).shuffle(index_array)

        copy_img_files =  list(map(copy_img_files._getitem_, index_array))
        copy_mask_files = list(map(copy_mask_files._getitem_, index_array))

        quantity_test = int(len(copy_img_files)*test_portion)

        test_set.img_files = copy_img_files[:quantity_test]
        test_set.mask_files = copy_mask_files[:quantity_test]

        train_set.img_files = copy_img_files[quantity_test:]
        train_set.mask_files = copy_mask_files[quantity_test:]
        return train_set,test_set

#To discuss, do we want to compose trasformations as well?
def augment_data(dataset,rootpath,rotation = False,verticalflip =False,horizontalflip = False, toGrayScale = False, colorJitter = False,combinations = 0,p = 1,nbr_rot = 1):
    """"Given a Dataloader and a output filepath takes all the images in the dataloader and augment them,
    then write the augmented images to the output filepath and returns the augmented dataloader"""
    #Define all the tranformers
    transformers = []
    processes = []
    #Verical mirroring
    if verticalflip:
    verticalFlip_ =  transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.RandomVerticalFlip(),
                               transforms.ToTensor()
    ])
    transformers.append(verticalFlip_)
    processes.append('VerticalFlip')
    #Horizontal mirroring
    if horizontalflip:
    horizontalFlip_ = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()
    ])
    transformers.append(horizontalFlip_)
    processes.append('HorizontalFlip')
    #set img to GrayScale
    if toGrayScale:
    grayScale_ = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.RandomGrayscale(p=p),
                               transforms.ToTensor()
    ])
    #Apply small random changes to input img to assure the model is noise agnostic
    if colorJitter:
    colorJitter_ = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.transforms.ColorJitter(),
                               transforms.ToTensor()
    ])
    if combinations != 0:
    combine_img = []
    combine_lbl = []
    for r in range(combinations):
        combine_img.append( transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(p=1),
                                transforms.RandomVerticalFlip(p=1),
                                transforms.RandomGrayscale(p=1),
                                transforms.transforms.ColorJitter(),
                                transforms.ToTensor()
        ])
        )
        combine_lbl.append( transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(p=1),
                                transforms.RandomVerticalFlip(p=1),
                                transforms.RandomGrayscale(p=1),
                                transforms.ToTensor()
        ]) 
        )
    #Apply trasformations and save the new immages
    print(len(dataset))
    for i in range(len(dataset)):
    img,lbl = dataset[i]
    #save grayScale pics (note: only grayscaling the img and not the label)
    if toGrayScale:
        #save also original dataset pics
        save_image(transforms.functional.to_tensor(transforms.functional.to_pil_image(img)),rootpath + '/images/'+'Original' + str(i) + '.png')
        save_image(transforms.functional.to_tensor(transforms.functional.to_pil_image(lbl)),rootpath + 'labels/'+'Original' + str(i) + '.png')
        #save grayscaled pics
        save_image(grayScale_(img),rootpath + '/images/'+'GrayScale' + str(i) + '.png')
        save_image(transforms.functional.to_tensor(transforms.functional.to_pil_image(lbl)),rootpath + 'labels/' +'GrayScale' + str(i) + '.png')
    #Save color jittered inputs with original outputs (note: only jittering the img and not the label)
    if colorJitter:
        #Save pics
        save_image(colorJitter_(img),rootpath + '/images/'+'ColorJitter' + str(i) + '.png')
        save_image(transforms.functional.to_tensor(transforms.functional.to_pil_image(lbl)),rootpath + 'labels/' +'ColorJitter' + str(i) + '.png')
    if rotation:
        for r in range(nbr_rot):
        #Rotations: (note: we need to rotate the img and lbl of the same angle)
        #generate rnd angle
        rand = torch.FloatTensor(1).uniform_(-180, 180)
        #Apply trasformation both on img and lbl
        newImg = transforms.functional.to_tensor(transforms.functional.to_pil_image(img))
        newImg = transform.functional.rotate(newImg,rand[0].item())
        newLbl = transforms.functional.to_tensor(transforms.functional.to_pil_image(lbl))
        newLbl = transform.functional.rotate(newLbl,rand[0].item())
        #Save pics
        save_image(newImg,rootpath + '/images/'+'RandRot' + str(i) + 'rotNb' +str(r) + '.png')
        save_image(newLbl,rootpath + 'labels/' +'RandRot' + str(i) + 'rotNb' +str(r) + '.png')
    if combinations != 0:
        #save combination of trasformations
        for r in range(combinations):
            save_image(combine_img[r](img),rootpath + '/images/'+'combine'+ str(i) +'combNb' +str(r) + '.png')
            save_image(combine_lbl[r](lbl),rootpath + 'labels/' +'combine' + str(i) +'combNb' +str(r) + '.png')
    #We need to perform the same trasformation to img and lbls therefore we concat them before the trasformation occurs
    seq = [img,lbl]
    concatenated = torch.cat(seq,axis = 0)
    trans = [transf(concatenated) for transf in transformers]
    #Once trasformation is performed we return to original format ("UNconcat")
    newImg = [tran[:3,:,:] for tran in trans]
    newLbl = [tran[3:,:,:] for tran in trans]
    #Save pics
    for j in range(len(transformers)):
        save_image(newImg[j],rootpath + '/images/'+processes[j] + str(i) + '.png')
        save_image(newLbl[j],rootpath + 'labels/' +processes[j] + str(i) + '.png')
    return DataLoaderSegmentation(rootpath)

# dataset = DataLoaderSegmentation(folder_path = TRAIN_FOLDER)
# train_set,test_set = dataset.split_train_test()

# newdatapath = '/content/drive/MyDrive/ML_PROJECT_2/training/NEWDATA/'
# for i in range(len(train_set)):
#   img,mask = train_set[i]
#   save_image(img, newdatapath +'TRAIN/'+'images/' + str(i) + '.png')
#   save_image(mask,newdatapath +'TRAIN/'+'labels/' + str(i) + '.png')
# for i in range(len(test_set)):
#   img,mask = test_set[i]
#   save_image(img,newdatapath +'TEST/'+'images/' + str(i) + '.png')
#   save_image(mask,newdatapath +'TEST/'+'labels/' + str(i) + '.png')

# pathbestmodel =  '/content/drive/MyDrive/ML_PROJECT_2/training/BESTMODELDATA/'
# TRAIN_FOLDER = '/content/drive/MyDrive/ML_PROJECT_2/training/'
# dataset = DataLoaderSegmentation(folder_path = TRAIN_FOLDER,images_folder = 'images',label_folder = 'groundtruth')
# augment_data(dataset,pathbestmodel,rotation = True,verticalflip =True,horizontalflip = True, toGrayScale = True, colorJitter = True,combinations = 10, nbr_rot = 10)

