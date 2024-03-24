import numpy as np
from os.path import join
import scipy.io as sio
import random
import mne


'''ch_names = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
            "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
            "O1", "O2"]'''

ch_names = ["Cz", "C3", "C4"]
event_id = {'left': 1,'right':2}

def get_data(subj,session,data_path):
    da=sio.loadmat(join(data_path,'sub-'+str(subj).zfill(3)+'_ses-'+str(session).zfill(2)+'_task_motorimagery_eeg.mat')) #da=sio.loadmat(join(data_path,'SHU_dataset','sub-'+str(subj).zfill(3)+'_ses-'+str(session).zfill(2)+'_task_motorimagery_eeg.mat'))
    data=da['data']
    labels=np.ravel(da['labels'])
    cz_index, c3_index, c4_index = 11, 12, 13
    selected_data = data[:,[cz_index, c3_index, c4_index], :]
    return selected_data,labels



def create_raw(data, labels, info, freq=250):
    labels = np.ravel(labels)
    ch_names = info.ch_names
    si, sj, sk = data.shape
    da = data.transpose(1, 0, 2)
    da = da.reshape(sj, si * sk)
    llen = data.shape[0]
    event = np.zeros((llen, 3))
    for i in range(llen):
        event[i, 0] = i * sk
        event[i, 2] = labels[i]
    event = event.astype(int)
    
    info = mne.create_info(ch_names=ch_names, ch_types="eeg", sfreq=freq)
    raw = mne.io.RawArray(da, info)  # create raw object
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    return raw, event


def mnebandFilter(data, labels, lowfreq, highfreq, freq=250):
    info = mne.create_info(ch_names=ch_names, ch_types='eeg', sfreq=freq)
    raw, event = create_raw(data, labels, info, freq)
    raw.filter(lowfreq, highfreq, fir_design='firwin')
    epochs = mne.Epochs(raw, event, event_id, 0, 4 - 0.004, baseline=None, preload=True)
    train_data = epochs.get_data()
    return train_data