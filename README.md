
# ML Project 2 - Team MNM - Option B - Road Segmentation

[AI Crowd Challenge link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation)

For this project task, we had to create a model to segment roads in satellite images, i.e assign a label road=1, background=0 to each pixel.

The dataset is available in this git aswell as on the CrowdAI.

Our best model achieved an F1-score of 90.7\% and an accuracy of 95\%.
Final submission: [Submission #169329](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/submissions/169329)

[Our Report](ML_MMN_Project2.pdf)

# How to reproduce our best submission

Clone this repo and follow the setup and run steps below!

## Environment Setup

Run the following commands to create an appropriate python environment and install all required libraries.

```shell
conda create -y -n ml_roadseg python=3.9.7 scipy pandas numpy matplotlib
conda activate ml_roadseg
pip install Pillow
pip install opencv-python
conda install -y pytorch torchvision torchaudio -c pytorch
```

## Running the code

```shell
# Activate python environment
conda activate ml_roadseg

# Run preprocessing/data augmentation
python run.py prepro

# Runs model training and saves model
python run.py train_model

# Loads trained model and runs predictions on test set
python run.py predict_test

# Read predicted labels and write them to the .csv submission format
python run.py write_sub
```
# Authors

Baldwin Nicolas - [chabala98](https://github.com/chabala98)

Leidi Mauro - [MauroLeidi](https://github.com/MauroLeidi)

Roust Michael - [michaelroust](https://github.com/michaelroust)
