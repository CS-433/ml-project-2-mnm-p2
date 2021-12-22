

import torch

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
import torchvision.transforms as transform
from torchvision.utils import save_image
import torch.functional as F
import PIL
import torch.utils.data as data
import glob
import random
from PIL import Image
import re

PREDICTION_DIR = 'data/predictions/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_images(img_folder,gt_folder,quantity,all = False):
    """load 'quantity images to 2 numpy array. satelite images and groundtruth images'"""

    #extract name of images in both folders
    files_img = os.listdir(img_folder)
    files_gt = os.listdir(gt_folder)

    files_img = sorted(files_img)
    files_gt = sorted(files_gt)
    #if we don't want all images extract only 'quantitiy' first images
    print(files_img[1])
    print(files_gt[1])
    if not all:
        files_img = files_img[:quantity]
        files_gt = files_gt[:quantity]

    #list containing the satelite images
    ls_img = []

    #extract satelite images
    i = 0
    for file in files_img:
        img = mpimg.imread(img_folder+file)
        ls_img.append(img)
        i+=1
        if i % 100 == 0:
          print(i)

    #list containing ground truths
    ls_gt = []
    i = 0
    #extract ground truths
    for file in files_gt:
        img = mpimg.imread(gt_folder+file)
        ls_gt.append(img)
        i+=1
        if i % 100 == 0:
          print(i)

    return np.asarray(ls_img),np.asarray(ls_gt)


def load_test_images(img_folder):
    """load 'quantity images to 2 numpy array. satelite images and groundtruth images'"""
    def get_number(x):
      return int(re.sub("[^0-9]", "", x))

    #extract name of images in both folders
    files_img = os.listdir(img_folder)
    files_img.sort(key = get_number)

    #list containing the satelite images
    ls_img = []

    #extract satelite images
    for file in files_img:
        print(file)
        img_path = img_folder+file+'/'+file + '.png'
        img = (transform.functional.pil_to_tensor(PIL.Image.open(img_path)).type(torch.FloatTensor)/255).unsqueeze(0)
        ls_img.append(img)

    return torch.cat(ls_img)


def turn_gt_to_torch_float(gt):
  """Transform groundtruth images to pytorch format"""
  return torch.from_numpy(np.round(gt)).type(torch.LongTensor).float()

def numpy_to_torch_format(images,gt):
  """Transform images and grountruth to torch format"""
  return torch.from_numpy(np.moveaxis(images,3,1)),turn_gt_to_torch_float(gt).unsqueeze(1)

def torch_to_numpy_format(images,gt):
  """Transform images and grountruth to numpy format"""
  return np.moveaxis(images.numpy(),1,3),gt.squeeze(1).numpy()

def train_model(model,train_set,test_set,optimizer,criterion,scheduler,
                filename,preprocess_img,preprocess_label,alpha = 0.5,betha = 0.5,mini_batch_size = 16,nb_epochs = 30, criterion2 = None,
                use_scheduler = True, print_progress= True,aux_loss = False, baseline_model = False):
    """Function to train models"""
    ### arguments:
    #             - model: model we want to train
    #             - train_set: dataloader containing the train set images and labels
    #             - test_set: dataloader containing the test set images and labels
    #             - optimizer: optimizer used for training(ex: ADAM)
    #             - criterion: loss function used for training.
    #                          If aux_loss = False, than provide your loss function here.
    #                          If aux_loss = True, criterion is the loss at the output of the real sized image (the 400x400)
    #             - scheduler: scheduler used for adjusting the learning rate during training
    #             - filename: name of file where we regularly save our model
    #             - preprocess_img: preprocess function to perform on an image before inserting it into the model (e.g: normalization)
    #             - preprocess_label: preprocess function to perform on a label
    #             - alpha: used when aux_loss = True, factor we multiply 'criterion' by when summing up the 2 loss functions
    #             - betha: used when aux_loss = True, factor we multiply 'criterion2' by when summing up the 2 loss functions
    #             - mini_batch_size: batch size
    #             - nb_epochs: number of epochs of training
    #             - criterion2: loss function used for training when aux_loss = True.
    #                          If aux_loss = False, criterion2 is not used
    #                          If aux_loss = True, criterion2 is the loss at the output of the pooled image ((400/16)x(400/16) = 25x25)
    #             - use scheduler: if True use a scheduler, if false don't
    #             - print_progress: if print progess, print statistics at the end of each epoch
    #             - aux_loss: if aux_loss = True, model uses an auxiliary loss

    #define parameters to save
    validation_loss = []
    train_loss = []
    validation_acc = []
    train_acc = []
    f1_score_val = []

    #put model in train mode
    model.train()
    batch_size = mini_batch_size
    #define pooling layer (for aux_loss = True)
    avg_pool = nn.AvgPool2d(kernel_size = (16,16), stride= (16,16), ceil_mode = False)
    for e in range(nb_epochs):
        #accumulated loss across epochs
        acc_loss = 0
        #determine current epoch on dataloader
        current_epoch = train_set.get_epoch()
        #variable that remembers the number of batchs of epoch
        nb_batches_train = 0
        #iterate train set for 1 epoch
        while train_set.get_epoch() == current_epoch:
            nb_batches_train += 1

            #get image and label from dataloader
            input,train_target = train_set.next(batch_size = batch_size)

            #preprocess image ( normalize)
            input = preprocess_img(input)

            #case where we use an auxiliary loss
            if aux_loss:
              #output is a 400x400 image and pooled_output is a 25x25 image (400x400 image average pooled)
              output,pooled_output = model(input.to(device))

              #calculate loss of first loss function
              loss = criterion(output,train_target.to(device))*alpha

              #average pool target to get 25x25 image
              pooled_train_target = avg_pool(train_target)

              #assign 0 or 1 values to target by thresholding
              pooled_train_target =  preprocess_label_basic_unet(pooled_train_target,threshold = 0.25)

              #calculate loss of second loss function
              loss2 = criterion2(pooled_output,pooled_train_target.to(device))*(betha)

              #sum up both losses
              loss += loss2

              #add to accumulated loss
              acc_loss = acc_loss + loss.item()

            #case where we use no aux loss
            else:
              #threshold target pixels to 0 or 1
              train_target = preprocess_label(train_target)

              #forward pass
              output = model(input.to(device))

              #calculate loss
              loss = criterion(output, train_target.to(device))

              #add to accumulated loss
              acc_loss = acc_loss + loss.item()

            #zero grad
            model.zero_grad()
            #backpropagate
            loss.backward()
            optimizer.step()

        ### validation statistics
        if(print_progress):
          ##loss

            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                nb_batches_test = 0
                current_epoch = test_set.get_epoch()

                ### SAME REASONING AS WITH TRAIN SET ABOVE
                while test_set.get_epoch() == current_epoch:
                  nb_batches_test += 1

                  input,test_target = test_set.next(batch_size = batch_size)

                  input = preprocess_img(input)

                  if aux_loss:
                    output,pooled_output = model(input.to(device))

                    loss = criterion(output, test_target.to(device))*alpha

                    pooled_test_target = avg_pool(test_target)
                    pooled_test_target = preprocess_label_basic_unet(pooled_test_target,threshold = 0.25)

                    loss += criterion2(pooled_output,pooled_test_target.to(device))*(betha)

                    val_loss = val_loss + loss.item()

                  else:

                    output = model(input.to(device))

                    test_target = preprocess_label(test_target)

                    loss = criterion(output, test_target.to(device))

                    val_loss = val_loss + loss.item()

                #print statistics
                print(e,'train_loss: ' ,acc_loss/nb_batches_train)
                train_loss.append(acc_loss/nb_batches_train)


                validation_loss.append(val_loss/nb_batches_test)
                print('val_loss: ', val_loss/nb_batches_test)
                f1_score = get_stats_images(model,test_set,5,preprocess_img,aux_loss = aux_loss)
                f1_score_val.append(f1_score)
                print('f1_score val:', f1_score)
                print()


        model.train()
        if use_scheduler:
          scheduler.step(f1_score_val[-1])
        torch.save(model.state_dict(), filename)
    return train_loss,validation_loss,train_acc,validation_acc

def compute_mean_std(dataset):
  """Function that computes the mean and std of a set of images in a dataloader"""

  img,_ = dataset.next(batch_size=1)
  #reset current idx of dataloader to 0
  dataset.currentidx = 0
  #get current epoch of dataset
  current_epoch = dataset.get_epoch()
  mean =0
  std = 0
  print('computing_mean')
  #iterate whole dataset (1 epoch)
  while dataset.get_epoch() == current_epoch:
    img,_ = dataset.next(batch_size=1)
    #add to current mean
    mean += img.mean()
  #compute mean
  mean /= len(dataset)
  print('computing_std')
  current_epoch = dataset.get_epoch()
  #iterate whole dataset (1 epoch)
  while dataset.get_epoch() == current_epoch:
    img,_ = dataset.next(batch_size=1)
    std += ((img*img).sum()-(mean*mean*400*400))
  #compute std
  std = torch.sqrt(std/(len(dataset)*400*400))
  return mean,std


def preprocess_label_basic_unet(image,threshold = 0.5):
  """threshold images to 0 or to 1"""
  image[image >= threshold] = 1
  image[image < threshold] = 0
  return image


def compute_pos_weight_matrix(dataset,aux_loss = False):
  """copmute positive weight matrix (for each pixel, compute (number pixels = 0) / (number pixels = 1))"""

  #if no aux loss, pos_weight matrix is a 400x400 images
  if not aux_loss:
    _,label = dataset.next(batch_size=1)
    dataset.currentidx = 0
    #get current epoch
    current_epoch = dataset.get_epoch()
    pos_nb = torch.zeros(label.shape)
    neg_nb = torch.zeros(label.shape)
    #iterate dataset
    while dataset.get_epoch() == current_epoch:
      _,label = dataset.next(batch_size=2)
      #threshold labels to 0 or 1
      label = preprocess_label_basic_unet(label)
      #compute number of pixels = 0
      neg_nb  += (label == 0).sum(dim = 0)
      #compute number of pixels = 1
      pos_nb += (label ==1).sum(dim = 0)
    #aggregate results
    pos_weight = neg_nb/pos_nb

  #if aux loss, pos_weight matrix is a 25x25 images
  else:
    #pooling which is equivalent to computing the mean of a patch
    avg_pool = nn.AvgPool2d(kernel_size = (16,16), stride= (16,16), ceil_mode = False)
    _,label = dataset.next(batch_size=1)
    label = avg_pool(label)
    dataset.currentidx = 0
    current_epoch = dataset.get_epoch()
    pos_nb = torch.zeros(label.shape)
    neg_nb = torch.zeros(label.shape)
    #iterate dataset
    while dataset.get_epoch() == current_epoch:
      _,label = dataset.next(batch_size=2)
      #downsize image to 25x25 (computing the mean in patches)
      label = avg_pool(label)
      #threshold label to 0 or1
      label = preprocess_label_basic_unet(label,threshold = 0.25)
      neg_nb  += (label == 0).sum(dim = 0)
      pos_nb += (label ==1).sum(dim = 0)
    #aggregate results
    pos_weight = neg_nb/pos_nb
  return pos_weight

def get_stats(pred,act):
  """provided the a prediction and it's labels, return TN,TP,FP,FN"""
  pred_0 = (pred == 0)
  pred_1 = (pred == 1)
  TN = (act[pred_0] == 0).sum()
  TP = (act[pred_1] == 1).sum()
  FN = (act[pred_0] == 1).sum()
  FP = (act[pred_1] == 0).sum()
  return TN,TP,FP,FN

def get_f1_score(tn,tp,fp,fn):
  """provided tn,tp,fp,fn , compute f1_score"""
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  if tp + fp == 0 or tp+fn == 0:
    return 0.
  return 2*precision*recall/(precision + recall)

def get_stats_images(model,data,batch_size,preprocess_img,aux_loss = False):
  """provided a dataloader with image and labels, compute f1 score of set"""

  model.eval()
  data.currentidx = 0
  current_epoch = data.get_epoch()
  avg_pool = nn.AvgPool2d(kernel_size = (16,16), stride= (16,16), ceil_mode = False)
  TN_tot = 0
  TP_tot = 0
  FP_tot = 0
  FN_tot = 0
  f1_scores = []
  #iterate dataset
  while data.get_epoch() == current_epoch:

    input,test_target = data.next(batch_size = batch_size)

    input = preprocess_img(input)
    if aux_loss:
      #compute f1 score with pooled output
      _,pooled_output = model(input.to(device))

      #threshold targets to 0 or 1
      pooled_test_target = avg_pool(test_target)
      pooled_test_target = preprocess_label_basic_unet(pooled_test_target,threshold = 0.25)

      #threshold predictions to 0 or 1 (threshold = 0.5 because pixel = 0.5 if the mean of the patch is 0.25)
      pooled_output = preprocess_label_basic_unet(pooled_output.detach().to('cpu'),threshold = 0.5)
      TN,TP,FP,FN = get_stats(pooled_output,pooled_test_target)

    else:
      #compute f1 score withoutput
      output = model(input.to(device))
      pooled_test_target = avg_pool(test_target)
      #threshold targets to 0 or 1
      pooled_test_target = preprocess_label_basic_unet(pooled_test_target,threshold = 0.25)
      pooled_output = avg_pool(output)
      #threshold predictions to 0 or 1
      pooled_output = preprocess_label_basic_unet(pooled_output.detach().to('cpu'),threshold = 0.25)
      TN,TP,FP,FN = get_stats(pooled_output,pooled_test_target)

    TN_tot += TN
    TP_tot += TP
    FP_tot += FP
    FN_tot += FN
  #return f1_score
  return get_f1_score(TN_tot,TP_tot,FP_tot,FN_tot)



def write_prediction(model,timg,preprocess_input,aux_loss = True):
  model.eval()
  for i in range(0,50):
    input = preprocess_input(timg[i].unsqueeze(0))
    if aux_loss:
      a,_ = model(input.to(device))
    else:
      a = model(input.to(device))

    a = a.to('cpu').detach()

    a,_ = torch_to_numpy_format(a,torch.zeros(1,608,608))

    a = a * 255
    if i < 9 :
      Image.fromarray(a[0,:,:,0].astype(np.uint8)).save(PREDICTION_DIR + "prediction_00" + str(i+1) + ".png")
    else:
      Image.fromarray(a[0,:,:,0].astype(np.uint8)).save(PREDICTION_DIR + "prediction_0" + str(i+1) + ".png")


"""
pipeline nicky for training best model:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_set = DataLoaderSegmentation(folder_path = NEW_DATA_FOLDER + 'TRAIN/AUGMENTED/',images_folder = 'images',label_folder = 'labels')
test_set = DataLoaderSegmentation(folder_path = NEW_DATA_FOLDER + 'TEST/',images_folder = 'images',label_folder = 'labels')

model = Unet_with_aux_loss_tanh(unet_pretrained_model)
model.to(device)

mean,std = compute_mean_std(train_set)

preprocess_input = transforms.Compose([transforms.Normalize(mean=mean, std=std)])


pos_weight = compute_pos_weight_matrix(train_set,aux_loss = True)

pos_weight= pos_weight.to(device)
criterion2 = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model.train()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
alpha = 0.5
betha = 0.5
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience = 4,threshold= 0.001,threshold_mode = 'abs',verbose = True,factor=0.5 )
train_loss,val_loss , _ , _ = train_model(model,train_set,test_set,optimizer,criterion,scheduler,
                                          '/content/drive/MyDrive/ml/results/best_model',preprocess_input,preprocess_label_basic_unet,alpha = alpha,betha = betha,mini_batch_size = 10,nb_epochs = 50, criterion2 = criterion2,
                                          use_scheduler = True, print_progress= True,aux_loss = True)
"""

