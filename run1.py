
from data_handling import *

TRAIN_FOLDER = 'data/'

dataset = DataLoaderSegmentation(folder_path = TRAIN_FOLDER,images_folder = 'images',label_folder = 'groundtruth')
train_set,test_set = dataset.split_train_test()

newdatapath = '/content/drive/MyDrive/ML_PROJECT_2/training/NEWDATA/'

for i in range(len(train_set)):
  img,mask = train_set[i]
  save_image(img, newdatapath +'TRAIN/'+'images/' + str(i) + '.png')
  save_image(mask,newdatapath +'TRAIN/'+'labels/' + str(i) + '.png')
for i in range(len(test_set)):
  img,mask = test_set[i]
  save_image(img,newdatapath +'TEST/'+'images/' + str(i) + '.png')
  save_image(mask,newdatapath +'TEST/'+'labels/' + str(i) + '.png')

pathbestmodel =  '/content/drive/MyDrive/ML_PROJECT_2/training/BESTMODELDATA/'
TRAIN_FOLDER = '/content/drive/MyDrive/ML_PROJECT_2/training/'

dataset = DataLoaderSegmentation(folder_path = TRAIN_FOLDER,images_folder = 'images',label_folder = 'groundtruth')
augment_data(dataset,pathbestmodel,rotation = True,verticalflip =True,horizontalflip = True, toGrayScale = True, colorJitter = True,combinations = 10, nbr_rot = 10)
