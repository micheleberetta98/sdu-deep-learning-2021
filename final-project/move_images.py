# %% Imports

import random
import shutil
import os
from pathlib import Path

# %% using random seed to randomize sample files
random.seed(0)

# %% Creating source and destination directories
source = Path('./images')
test_dest = Path('./testing')
val_dest = Path('./validation')
train_dest = Path('./training')

for d in [test_dest, val_dest, train_dest]:
    os.mkdir(d)
    os.mkdir(d / 'normal')
    os.mkdir(d / 'pneumonia')

# %% File copying
def copy_random(no_of_files, status, dest):
    all_files = [x for x in os.listdir(source) if status in x]
    random_filenames = random.sample(all_files, no_of_files)
    for filename in random_filenames:
        shutil.move(source / filename, dest / status / filename)


print('Moving 900 files to training... ', end='')
copy_random(900, "normal", train_dest)
copy_random(900, "pneumonia", train_dest)
print('Done')

print('Moving 100 files to testing... ', end='')
copy_random(100, "normal", test_dest)
copy_random(100, "pneumonia", test_dest)
print('Done')

print('Moving 100 files to validation... ', end='')
copy_random(100, "normal", val_dest)
copy_random(100, "pneumonia", val_dest)
print('Done')
