import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torch

class EEGFeatureDataset(Dataset):
    def __init__(self, data_dir, feature_dir, subjects, sessions, feature_fold=10):
        #self.data_samples = []
        #self.feature_samples = []

        for subj in subjects:
            for sess in sessions:
                data_path = os.path.join(data_dir, f'sub-{subj:03d}_ses-{sess:02d}_task_motorimagery_eeg.mat')
                feature_path = os.path.join(feature_dir, f'sub_{subj}_ses_{sess}_fold_{feature_fold}_train_features.npy')

                # Load EEG data
                mat_data = sio.loadmat(data_path)
                all_data = mat_data['data'] 
                cz_index, c3_index, c4_index = 11, 12, 13
                data = all_data[:,[cz_index, c3_index, c4_index], :] 

                # Load features
                features = np.load(feature_path)

                min_trials = min(data.shape[0], features.shape[0])
                eeg_shape = data.shape[1] * data.shape[2]
                features_no = features.shape[1]
                self.data_samples = np.ones((min_trials,eeg_shape))
                self.feature_samples = np.ones((min_trials,features_no))

                for trial in range(min_trials):
                    flattened_data = data[trial].flatten()
                    self.data_samples[trial,:] = flattened_data
                    self.feature_samples[trial,:] = features[trial,:]

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        data_sample = torch.tensor(self.data_samples[idx], dtype=torch.float32)  # Convert to float32
        feature_sample = torch.tensor(self.feature_samples[idx], dtype=torch.float32)  # Convert to float32
        return data_sample, feature_sample

data_dir = '/Users/umairarshad/SHU/or_data/mat'
feature_dir = '/Users/umairarshad/SHU/or_data/features'  
subjects = range(6, 11)  
sessions = range(1, 6)  

dataset = EEGFeatureDataset(data_dir, feature_dir, subjects, sessions)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

type(dataset)

"""mat_data = sio.loadmat('/Users/umairarshad/SHU/or_data/mat/sub-001_ses-01_task_motorimagery_eeg.mat')
data = mat_data['data']
print(data.shape)

features = np.load(feature_dir + '/sub_1_ses_1_fold_1_train_features.npy')
min_trials = min(data.shape[0], features.shape[0]) 
data_samples_check = np.zeros((min_trials,32000))#[]
features_samples_check = np.zeros((min_trials,10))#[]

for trial in range(min_trials):
    flattened_data = data[trial].flatten()
    data_samples_check[trial,:] = flattened_data
    features_samples_check[trial, :] = features[trial,:]
    #data_samples_check.append(flattened_data)
    #features_samples_check.append(features[trial])
print(flattened_data.shape)
print(data_samples_check.shape)
print(features_samples_check.shape)"""