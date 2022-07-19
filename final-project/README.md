# Final Project - Group 2

## Organization

There are various files:
* `constants.py`, which contains some useful constats
* `image_generation.py`, which contains all the `ImageDataGenerator`s (author: Mark John Paul Pangan)
* `model_hyperpameters.py`, used to test a couple of hyperparameters for the convolutional part (authors: Bianca Crippa, Michele Beretta)
* `model.py`, which contains the final model (authors: Bianca Crippa, Michele Beretta)
* `move_images.`, a script to split the images into the various sets (author: Narendro Ariwisnu)
* `visualizations.py`, which contains all the code to do the visualizations (author: Tanja Gurtner)

Of course, the author of a file is the person that has put the most in it, but we all
cooperated on almost all the tasks, especially in the network definition and testing.

## Models

Under the folder `final-model` there is the final model we want to submit, with its specific visualizations.

The history visualizations were done with *Weight and Biases*, and the images provided are the export of those.
You can see all the plots [here](https://wandb.ai/micheleberetta98/sdu-deep-learning-final?workspace=user-micheleberetta98).

## About multiprocessing

We used `use_multiprocessing=True` and `workers=8` in the fitting process.
This could cause some problems if you want to run the project on your computer, so you should remove these two
lines from all `model.fit` calls (or just adapt them using your processor specifications).

## GitHub repo

The code in its entirety is available [at this GitHub repo](https://github.com/micheleberetta98/sdu-deep-learning-2021/tree/master/final-project).

## Images specifications

Our image's size is 256x256, and the images have been converted to grayscale.

## How to move the images into the correct folders

1. Extract the files in `data.zip` into an `images` folder
2. Launch `python move_images.py` from the console