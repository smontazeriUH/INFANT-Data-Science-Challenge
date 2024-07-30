""" 01_preprocess.py
# This piece of code reads raw EEG files stored in "EDF_format" folder and HIE 
# grades from "eeg_grades.csv", creates four bipolar couplings (F3-C3, F4-C4, 
# T3-O1, T4-O2), filters with a bandpass (.5-32Hz) and a 50Hz notch filter,  
# resamples and segments the data to be ready for the next stages in 
# "02_readfiles.py" and "04_readfiles-test.py"
# outputs will be save into "train_64_npy" and "tests_64_npy" folders. Please 
# check if the folders exist otherwise please make them."""

import mne
import csv
import numpy as np
import time
import os
import zipfile
from collections import defaultdict
from scipy.ndimage.filters import uniform_filter1d

import scipy.signal as signal
from tensorflow.keras.models import load_model
import onnxruntime as rt # onnx runtime is the bit that deals with the ONNX network

import pandas as pd

eeg_df = pd.read_csv("../eeg_grades.csv")
grades = eeg_df['grade']
grades = grades.fillna(0)

active = ['F3', 'F4', 'T3', 'T4']
reference = ['C3', 'C4', 'O1', 'O2']

def loadFile(inputaddress):
    if '.edf' in inputaddress:
        file_data = f'{inputaddress}'
    
        data = mne.io.read_raw_edf(file_data)
        fs = data.info['sfreq']
        allChName = data.ch_names
        
        allsampleRates = np.float64(fs)*np.ones((5,))
        returnedDataAll, times = data.get_data(picks=['eeg'], units = {'eeg': 'uV'}, return_times=True, start = 0, stop = (60*60*int(fs)))
        
        return returnedDataAll, allsampleRates, allChName, np.shape(returnedDataAll)[1]/fs


filenames = []
Groups = []
labels = []

train_data_dir = '../EDF_format/'
for subdir, dirs, files in os.walk(train_data_dir):
    files = [f for f in files if not f[0] == '.']
    for file in sorted(files):
            filenames.append(subdir+file)
            Groups.append(file[2:4])
   
print(len(filenames))

for i in range(len(filenames)):
    rawEEG, fs, channels, duration = loadFile(filenames[i])
    len1 = int(fs[0]*duration)
    b, a = signal.cheby2(6, 80, [0.5/(fs[0]/2), 32/(fs[0]/2)], 'bandpass', output = 'ba') # design bandpass filter
    bn, an = signal.iirnotch(50,30,fs[0]) # design 50Hz notch filter
    # Bipolar montage data
    cc = np.zeros((len(active),len1))
    for ii in range(0,len(active)):    
        index1 = np.where(np.array(channels) == active[ii])[0][0]
        index2 = np.where(np.array(channels) == reference[ii])[0][0]
        cc[ii,:] = rawEEG[index1,:] - rawEEG[index2,:]
    
    HIEG = int(grades[i])
    
    # This is the implementation of the NN using ONNX (note you have to do extra things to install ONNX but it is easy and on the web)
    mini_block_no = duration/30
    n = len(active) # number of samples
    c = 1 # rgb colour depth
    h = 3840 # epoch length
    w = 1 # 1D signal
    # build the data tensor to feed into the network so take an EEG segment first and then channel to form tensor
    for ii in range(0, int(mini_block_no)):
        r1 = int(ii*fs[0]*30)
        r2 = r1+int(60*fs[0])
        dd = cc[:,r1:r2]
        aa = np.zeros((h,n)).astype('float32')
        for jj in range(0,n):
            epoch = dd[jj,:]
            epoch = signal.lfilter(bn,an, epoch) # 50Hz notch pass filter
            epoch = signal.lfilter(b,a, epoch) # bandpass filter need to add 50Hz notch 
            epoch = signal.resample(epoch, h)  #  this is going from 64 to 32Hz (check)
            epoch[epoch>250] = 250 
            epoch[epoch<-250]=-250
            epoch = epoch/251
            epoch = epoch.astype('float32')
            aa[:,jj] = epoch
        print('../train_64_npy/' + filenames[i][14:-4] + '_' + str(ii+10).zfill(3) + '.npy')
        if HIEG != 0:
            os.chdir('../train_64_npy/' )
            np.save('../train_64_npy/' + filenames[i][14:-4] + '_' + str(ii+10).zfill(3) + '_' + str(1) + '_' + str(HIEG-1) + '.npy', aa[:,0])
            np.save('../train_64_npy/' + filenames[i][14:-4] + '_' + str(ii+10).zfill(3) + '_' + str(2) + '_' + str(HIEG-1) + '.npy', aa[:,1])
            np.save('../train_64_npy/' + filenames[i][14:-4] + '_' + str(ii+10).zfill(3) + '_' + str(3) + '_' + str(HIEG-1) + '.npy', aa[:,2])
            np.save('../train_64_npy/' + filenames[i][14:-4] + '_' + str(ii+10).zfill(3) + '_' + str(4) + '_' + str(HIEG-1) + '.npy', aa[:,3])
        else:
            os.chdir('../tests_64_npy/' )
            np.save('../tests_64_npy/' + filenames[i][14:-4] + '_' + str(ii+10).zfill(3) + '_' + str(6) + '.npy', aa)