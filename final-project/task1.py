# -*- coding: utf-8 -*-
"""
Task 1
"""

import numpy as np
import os
import random

import shutil

#%% set directories
directory = r'C:\Users\Admin\OneDrive\ZHAW\17_DeepLearning\Project\data'

#%% get files

pneumonia_files = [f for f in os.listdir(directory) if f.endswith('pneumonia.jpg')]
normal_files = [f for f in os.listdir(directory) if f.endswith('normal.jpg')]

#%% assign files

list_pneumonia = list(range(1, 1100))
list_normal= list(range(1, 1100))

train_pneumonia = random.sample(list_pneumonia, 440)
train_normal = random.sample(list_normal, 440)

list_pneumonia = [x for x in list_pneumonia if x not in train_pneumonia]
list_normal = [x for x in list_normal if x not in train_normal]

test_pneumonia = random.sample(list_pneumonia, 55)
test_normal = random.sample(list_normal, 55)

list_pneumonia = [x for x in list_pneumonia if x not in test_pneumonia]
list_normal = [x for x in list_normal if x not in test_normal]

val_pneumonia = random.sample(list_pneumonia, 55)
val_normal = random.sample(list_normal, 55)

#%% create file name
pneumonia = "_pneumonia.jpg"
normal = "_normal.jpg"

train_pneumonia = [str(number) + pneumonia for number in train_pneumonia]
test_pneumonia = [str(number) + pneumonia for number in test_pneumonia]
val_pneumonia = [str(number) + pneumonia for number in val_pneumonia]

train_normal = [str(number) + normal for number in train_normal]
test_normal = [str(number) + normal for number in test_normal]
val_normal = [str(number) + normal for number in val_normal]



# %% copy / move files

#shutil.move

for f in train_pneumonia:
    try:
        shutil.copy(f, r'C:\Users\Admin\OneDrive\ZHAW\17_DeepLearning\Project\Training\pneumonia')
    except:
        continue
    
for f in train_normal:
    try:
        shutil.copy(f, r'C:\Users\Admin\OneDrive\ZHAW\17_DeepLearning\Project\Training\normal')
    except:
        continue
    
for f in val_pneumonia:
    try:
        shutil.copy(f, r'C:\Users\Admin\OneDrive\ZHAW\17_DeepLearning\Project\Validation\pneumonia')
    except:
        continue
    
for f in val_normal:
    try:
        shutil.copy(f, r'C:\Users\Admin\OneDrive\ZHAW\17_DeepLearning\Project\Validation\normal')
    except:
        continue

for f in test_pneumonia:
    try:
        shutil.copy(f, r'C:\Users\Admin\OneDrive\ZHAW\17_DeepLearning\Project\Testing\pneumonia')
    except:
        continue
    
for f in test_normal:
    try:
        shutil.copy(f, r'C:\Users\Admin\OneDrive\ZHAW\17_DeepLearning\Project\Testing\normal')
    except:
        continue
