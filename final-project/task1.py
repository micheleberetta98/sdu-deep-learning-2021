# -*- coding: utf-8 -*-
"""
Task 1
"""
# %%

import numpy as np
import os
import random
from pathlib import Path

import shutil

random.seed(0)

# %% set directories
directory = Path('./images')

training_dir = Path('./training')
testing_dir = Path('./testing')
validation_dir = Path('./validation')

# %% get files


pneumonia_files = [f for f in os.listdir(
    directory) if f.endswith('pneumonia.jpg')]
normal_files = [f for f in os.listdir(directory) if f.endswith('normal.jpg')]

# %% assign files

list_pneumonia = list(range(1, 1100 + 1))
list_normal = list(range(1, 1100 + 1))

train_pneumonia = random.sample(list_pneumonia, 900)
train_normal = random.sample(list_normal, 900)

list_pneumonia = [x for x in list_pneumonia if x not in train_pneumonia]
list_normal = [x for x in list_normal if x not in train_normal]

test_pneumonia = random.sample(list_pneumonia, 100)
test_normal = random.sample(list_normal, 100)

list_pneumonia = [x for x in list_pneumonia if x not in test_pneumonia]
list_normal = [x for x in list_normal if x not in test_normal]

val_pneumonia = random.sample(list_pneumonia, 100)
val_normal = random.sample(list_normal, 100)

# %% create file name
pneumonia = "_pneumonia.jpg"
normal = "_normal.jpg"

train_pneumonia = [directory / (str(number) + pneumonia)
                   for number in train_pneumonia]
test_pneumonia = [directory / (str(number) + pneumonia)
                  for number in test_pneumonia]
val_pneumonia = [directory / (str(number) + pneumonia)
                 for number in val_pneumonia]

train_normal = [directory / (str(number) + normal) for number in train_normal]
test_normal = [directory / (str(number) + normal) for number in test_normal]
val_normal = [directory / (str(number) + normal) for number in val_normal]


# %% copy / move files

# shutil.move

for f in train_pneumonia:
    shutil.copy(f, training_dir / 'pneumonia')

for f in train_normal:
    shutil.copy(f, training_dir / 'normal')

for f in val_pneumonia:
    shutil.copy(f, validation_dir / 'pneumonia')

for f in val_normal:
    shutil.copy(f, validation_dir / 'normal')

for f in test_pneumonia:
    shutil.copy(f, testing_dir / 'pneumonia')

for f in test_normal:
    shutil.copy(f, testing_dir / 'normal')
