
# ML Project 2 - Team MNM - Road Segmentation

[Crowd AI Challenge link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation)

For this project task, we had to create a model to segment roads in satellite images, i.e assign a label road=1, background=0 to each pixel.

The dataset is available in this git aswell as on the CrowdAI.

Evaluation Metric: [F1 score](https://en.wikipedia.org/wiki/F-score)

## How to reproduce our best submission

Follow the steps in Environment Setup to be able to run the code.

Then just execute run.py

## Environment Setup

Run the following commands to create an appropriate python environment and install all required libraries.

```shell
conda create -y -n ml_roadseg python=3.9.7 scipy pandas numpy matplotlib
conda activate ml_roadseg
pip install Pillow
conda install -y pytorch torchvision torchaudio -c pytorch
```
