"""
    This file is a script to move the images into the correct folders
    @author Narendro Ariwisnu
"""

# %% Imports

import random
import shutil
import os
from pathlib import Path

from constants import NUM_OF_IMAGES_TEST, NUM_OF_IMAGES_TRAIN, NUM_OF_IMAGES_VAL

# Here we force the seed to a definite value so that the images split into training/validation/testing
# is reproducible

# We decidet to split the images as follow
# - 1800 for training (~80%)
# - 200 for testing (~10%)
# - 200 for testing (~10%)

random.seed(0)

# %% Defining source and destination directory
source = Path('./images')
test_dest = Path('./testing')
val_dest = Path('./validation')
train_dest = Path('./training')

# %% Creating normal and pneumonia directories inside destination directories for each testing, validation and training directory
for d in [test_dest, val_dest, train_dest]:
    os.mkdir(d)
    os.mkdir(d / 'normal')
    os.mkdir(d / 'pneumonia')

# %% Function to move files


def copy_random(no_of_files, status, dest):
    all_files = [x for x in os.listdir(source) if status in x]
    random_filenames = random.sample(all_files, no_of_files)
    for filename in random_filenames:
        shutil.move(source / filename, dest / status / filename)


# %% Moving files to training directory
print('Moving 900 files to training... ', end='')
copy_random(NUM_OF_IMAGES_TRAIN // 2, "normal", train_dest)
copy_random(NUM_OF_IMAGES_TRAIN // 2, "pneumonia", train_dest)
print('Done')

# %% Moving files to testing directory
print('Moving 100 files to testing... ', end='')
copy_random(NUM_OF_IMAGES_TEST // 2, "normal", test_dest)
copy_random(NUM_OF_IMAGES_TEST // 2, "pneumonia", test_dest)
print('Done')

# %% Moving files to validation directory
print('Moving 100 files to validation... ', end='')
copy_random(NUM_OF_IMAGES_VAL // 2, "normal", val_dest)
copy_random(NUM_OF_IMAGES_VAL // 2, "pneumonia", val_dest)
print('Done')
