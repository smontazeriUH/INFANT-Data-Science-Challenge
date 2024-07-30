""" 05_testCNN.py
# This piece of code tests the CNN model with the test data. It reads
# filenames of the test files stored in "validation_set.csv" and classifier
# models from "classifiers" folder. The output which is the classification
# results will be save into "my_submission.csv" file. next stage is performance
# evaluation.

"""

import scipy
import scipy.signal
import scipy.io
import scipy.stats

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from collections import Counter


def find_majority(votes):
    # Function to find majority voted class
    votes = votes[votes != 7]
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
        # print('It is a tie')
        if top_two[0][0] <= top_two[1][0]:
            return top_two[1][0]
        else:
            return top_two[0][0]
    return top_two[0][0]


X_test_filenames = np.load('./test_list/X_test_filenames.npy')
validation_df = pd.read_csv('../validation_set.csv')

cls_predictions = np.empty(shape=(len(validation_df), 3))

for cls_num in range(0, 3):
    print(cls_num)
    loadAddress = "./classifiers/model_%s.h5" %(cls_num+1)
    model = load_model(loadAddress, compile=False)
    # Initialise a list to store the predictions
    predIdxs_hours = np.empty(shape = (0,))

    for i in range(len(validation_df)):
        # Locate 1-min EEG segments corresponding to each 1-h recordings
        subfilenames_in_list = [string for string in X_test_filenames if 
                                validation_df.ID[i] in string]
        # Initialise a list to store the predictions
        predIdxs_tmp = np.empty(shape = (0,))
        for j in range(len(subfilenames_in_list)):
            # Read an EEG segment               
            dataloaded = np.load('../tests_64_npy/' + subfilenames_in_list[j])
            
            if (dataloaded.size != 0):
                # Initialise and read each EEG channel
                X1 = np.empty((1,3840,1))
                X2 = np.empty((1,3840,1))
                X3 = np.empty((1,3840,1))
                X4 = np.empty((1,3840,1))
                X1[0,:,0] = dataloaded[:,0]
                X2[0,:,0] = dataloaded[:,1]
                X3[0,:,0] = dataloaded[:,2]
                X4[0,:,0] = dataloaded[:,3]
                
                # Make predictions
                predictedValues1 = model.predict(X1, steps=1, verbose=0)
                predictedValues2 = model.predict(X2, steps=1, verbose=0)
                predictedValues3 = model.predict(X3, steps=1, verbose=0)
                predictedValues4 = model.predict(X4, steps=1, verbose=0)
                
                # Average over channels
                MEanpredictedValues = np.mean([predictedValues1,
                                               predictedValues2,
                                               predictedValues3,
                                               predictedValues4],axis=0)
                # Class with maximun occurance
                predIdxs = np.argmax(MEanpredictedValues)
            else:
                predIdxs = 7 # an arbitraty label to indicate unsuccessful reading of the file
                
            # Store per-minute results 
            predIdxs_tmp = np.concatenate((predIdxs_tmp, [predIdxs])) # store per-minute results

        # Combine per-minute results to get per-hour results
        predIdxs_hours = np.concatenate((predIdxs_hours,[find_majority(scipy.signal.medfilt(predIdxs_tmp, kernel_size = 7))]))

    # Modify class indices
    cls_predictions[:,cls_num] = predIdxs_hours+1

# Find majority class over classifiers
final_results = scipy.stats.mode(cls_predictions.T, axis=0).mode[0]
# Create a dataset that matches the index of the data used as validation:
pred_df = pd.DataFrame({'class': final_results}, index=validation_df.index)
# Match the integer datatype:
pred_df['class'] = pred_df['class'].astype('int8')
print(pred_df.head())

pred_df.to_csv('../my_submission.csv')
