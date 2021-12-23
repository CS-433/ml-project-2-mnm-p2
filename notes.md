
## Setup

```shell
conda create -y -n ml scipy pandas numpy matplotlib
conda activate ml
conda install -y pytorch torchvision torchaudio -c pytorch
```

## run.py

Should do:
- Preprocess data
- Run model training
- Run predictions on test_images
- Output predictions

## For Nicky

Some context:
The code Nicky (you) sent me (in the afternoon) is in `model_training.py`.
The code Mauro sent is in `data_handling.py`.
I didn't run stuff from `model_training.py` (I'll get back to that later)
but everything in `data_handling.py` should now be working smoothly!


Take a look at the following files
- `helpers.py` you can swap a line of code to make it work on google colab.
You would just have to copy the file structure that currently exists in the repo. I would
just copy everything into google drive and then run stuff from `run.py` using "!".
Example: to run preprocessing: `!python run.py prepro`
- `run.py` basically the file we need to work on. I put some TODO comments for you!
- `predicting_images.py` what you sent me. It should be called from `run.py`
but I couldn't test it (roughly speaking no GPU device API for mac).
- `README.md` if you have any issues with some packages or using `run.py`

Before you start working just run
`python run.py prepro` or `!python run.py prepro` if ur doing it in a notebook.
it will generate all the augmented data. Then you just have to fill the 1st TODO
in `run.py` and run.
