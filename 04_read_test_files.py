""" 04_read_test_files.py
# This piece of code reads preprocessed EEG files from "tests_64_npy" folder
# and only stores filenames into a numpy file
# output will be save into "test_list" folder. Please check if the folder exists
# otherwise please make it.
# next stage is testing the CNN. Please go to "05_testCNN.py"."""

import scipy
from scipy import io
import random

import os
import shutil

import numpy as np

import keras
from tensorflow.keras.utils import to_categorical

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, ShuffleSplit
from sklearn.metrics import confusion_matrix
train_data_dir = '../tests_64_npy/'

save_to = './test_list/'

filenames = []
Groups = []
labels = []
ARFilenames = []
ARGroups = []
ARLabels = []
for subdir, dirs, files in os.walk(train_data_dir):
    files = sorted([f for f in files if not f[0] == '.'])
    dirs[:] = [d for d in dirs if not d[0] == '.']
    for file in files:
            filenames.append(file)
            Groups.append(file[2:4])
            labels.append(int(file[-5]))
    
print(len(filenames))

# One hot vector representation of labels
y_labels_one_hot = to_categorical(labels, dtype='int32')
filenames_numpy = np.array(filenames)

np.save(save_to + 'X_test_filenames.npy', np.asarray(filenames_numpy, dtype=np.str))
print(len(filenames)) # 
    # i = i + 1