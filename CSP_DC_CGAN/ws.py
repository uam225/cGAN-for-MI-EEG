'''feature extractor'''
import numpy as np
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from data_preprocess import get_data, mnebandFilter
import os

def train_test(sub_id, test_session, data_path, k_fold=10):
    data, labels = get_data(sub_id, test_session, data_path)
    data = mnebandFilter(data, labels, 3, 35)
    cv = StratifiedKFold(k_fold, shuffle=False)
    cv_split = cv.split(data, labels)
    scores = []

    fold = 0
    for i, j in cv_split:
        fold += 1
        trainData = data[i]
        trainLabels = labels[i]
        valData = data[j]
        valLabels = labels[j]

        csp = CSP(n_components=10, reg=None, log=False, norm_trace=False)
        trainFeature = csp.fit_transform(trainData, trainLabels)
        valFeature = csp.transform(valData)

        lda = LinearDiscriminantAnalysis()
        lda.fit(trainFeature, trainLabels)
        scores.append([lda.score(valFeature, valLabels)])

        # Save the features for each fold separately
        feature_file = os.path.join(data_path, f'sub_{sub_id}_ses_{test_session}_fold_{fold}_train_features.npy')
        np.save(feature_file, trainFeature)
        feature_file = os.path.join(data_path, f'sub_{sub_id}_ses_{test_session}_fold_{fold}_val_features.npy')
        np.save(feature_file, valFeature)

    aver_score = np.average(scores)
    return aver_score

import numpy as np

# Replace 'feature_file_path' with the actual path to one of your saved feature files
feature_file_path = '/Users/umairarshad/SHU/or_data/features/sub_6_ses_1_fold_1_train_features.npy'

# Load the features from the file
features = np.load(feature_file_path)

# Print the shape of the array to understand its dimensions
print("Shape of the features array:", features.shape)

# Print the data type of the array
print("Data type of the features array:", features.dtype)

# Print the first few elements to see a sample of the data
print("First few features:")
print(features[:5])

'''import mne
import scipy.io
import numpy as np

# Path to your MATLAB file
mat_file_path = '/Users/umairarshad/SHU/or_data/mat/sub-001_ses-01_task_motorimagery_eeg.mat'

# Load your MATLAB file
mat_data = scipy.io.loadmat(mat_file_path)

# Assuming 'data' is the key in your MATLAB file that contains the EEG data
# and it's structured as (n_channels, n_times) or (n_channels, n_times, n_epochs)
eeg_data = mat_data['data']

# If your data includes epochs, you might need to reshape or select a single epoch for visualization
# e.g., eeg_data = eeg_data[:, :, 0] for the first epoch if the data includes epochs

# Define channel types and names if known
channel_types = ['eeg'] * eeg_data.shape[0]  # Replace 'eeg' with the correct type if needed
channel_names = ['Ch{}'.format(i+1) for i in range(eeg_data.shape[0])]  # Example channel names

# Create an MNE Info object (replace 1000 with your actual sampling frequency)
# Assuming eeg_data shape is (n_epochs, n_channels, n_samples)
n_epochs, n_channels, n_samples = eeg_data.shape

# Reshape the data to (n_channels, n_epochs * n_samples)
eeg_data_reshaped = eeg_data.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_samples)

# Now create the MNE Raw object with the reshaped data
info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types=channel_types)
raw = mne.io.RawArray(eeg_data_reshaped, info)

# Print and plot the data
print(raw)
raw.plot()
'''

