
import sys

from helpers import *

from data_handling import *
from model_training import *
from mask_to_submission import *

from predicting_images import *

#=============================================================================


def run_prepro():
    """
    Using raw training files generates all files that will be used for training
    our model. The main part of this function to run all data augmentation.
    """

    create_missing_directories()
    print("Starting data preprocessing/augmentation")
    dataset = DataLoaderSegmentation(root_path = RAW_TRAIN_DATA_FOLDER,images_folder = 'images',label_folder = 'groundtruth')
    augment_data(dataset,FINAL_DATA_FOLDER,rotation = True,verticalflip =True,horizontalflip = True, toGrayScale = True, colorJitter = True,combinations = 10, nbr_rot = 10)
    print("Done")


def run_model_training():
    """
    Using preprocessed data to train the model and save model to a file
    """
    #load data
    augmented_dataset = DataLoaderSegmentation(root_path = FINAL_DATA_FOLDER,images_folder = 'images',label_folder = 'labels')
    train_set,test_set = augmented_dataset.split_train_test(test_portion=0.1, seed=1)

    #initialize model
    unet_pretrained_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model = Unet_with_aux_loss_tanh(unet_pretrained_model)

    #compute mean and std for normalization
    mean,std = compute_mean_std(train_set)
    preprocess_input = transforms.Compose([transforms.Normalize(mean=mean, std=std)])

    #compute positive label weights for BCEWithLogitsLoss
    pos_weight = compute_pos_weight_matrix(train_set,aux_loss = True)
    pos_weight= pos_weight.to(device)
    criterion2 = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)

    #weights of each loss function: loss = criterion*alpha + criterion2*betha
    alpha = 0.5
    betha = 0.5

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience = 4,threshold= 0.0001,threshold_mode = 'abs',verbose = True,factor=0.5 )
    train_loss,val_loss , _ , _ = train_model(model,train_set,test_set,optimizer,criterion,scheduler,
                                            MODELS_FOLDER + 'best_model.zip',preprocess_input,preprocess_label_basic_unet,alpha = alpha,betha = betha,mini_batch_size = 10,nb_epochs = 100, criterion2 = criterion2,
                                            use_scheduler = True, print_progress= True,aux_loss = True)



def run_predict_test():
    """
    Using saved model to run prediction generation on test set files.
    """

    #NOTE TO USER: IF YOU HAVE TRAINED YOUR MODEL, replace the final argument of this function by 'models/best_model.zip'
                #  otherwise just use the model we trained ('models/best_model_alpha05.zip')!
    predict_with_best_model(RAW_TEST_DATA_FOLDER, RAW_TEST_PREDICTIONS_FOLDER, 'models/best_model_alpha05.zip')



def run_write_test_to_submission():
    """
    Function writes test set predictions to .csv submission format.
    """
    submission_filename = 'best_model_submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = RAW_TEST_PREDICTIONS_FOLDER + 'prediction_' + '%.3d' % i + '.png'
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)


#=============================================================================

if __name__ == "__main__":
    print(len(sys.argv))
    print(sys.argv)

    mode = sys.argv[1]

    if mode == 'prepro':
        run_prepro()
    elif mode == 'train_model':
        run_model_training()
    elif mode == 'predict_test':
        run_predict_test()
    elif mode == 'write_sub':
        run_write_test_to_submission()
    else:
        print('Passed invalid mode argument')

