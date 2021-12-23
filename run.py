
import sys

from helpers import *

from data_handling import *
from model_training import *

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

    augmented_dataset = DataLoaderSegmentation(root_path = FINAL_DATA_FOLDER,images_folder = 'images',label_folder = 'labels')
    train_set,test_set = augmented_dataset.split_train_test(test_portion=0.1, seed=1)

    # Since we are using a set seed for spliting the result of the splitting should always be the same.
    # TODO NICKY please finish this function. Train the model and save it to a file.
    # as I understand you need test_set to be able to slow down the learning rate when needed so here it is (above).



def run_predict_test():
    """
    Using saved model to run prediction generation on test set files.
    """

    # TODO NICKY I couldn't make it run on my PC cause of weird MacOS specific
    # pytorch library shit
    # So make it run with the GPU plz.
    predict_with_best_model(RAW_TEST_DATA_FOLDER, RAW_TEST_PREDICTIONS_FOLDER, 'models/best_model_alpha05.zip')



def run_write_test_to_submission():
    """
    Function writes test set predictions to .csv submission format.
    """

    # TODO NICKY just make it read images from RAW_TEST_PRECITIONS_FOLDER and write to .csv
    pass


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

