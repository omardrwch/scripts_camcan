"""
Save samples of EOG time series
"""

import numpy as np
import camcan_utils
import mfanalysis as mf
import mf_config
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os.path as op
import mne
import os
import h5py

# MF parameters
param = {}
param['wt_name']   = 'db3'
param['formalism'] = None
param['p']         = np.inf
param['j1']        = 9
param['j2']        = 13
param['n_cumul']   = 3
param['gamint']    = 0.0
param['wtype']     = 0
mfa = mf.MFA(**param)

# Subjects and conditions
subjects = camcan_utils.subjects
conditions = camcan_utils.kinds

subject = subjects[0]

# MF parameters
mf_params = mf_config.get_mf_params()


def get_eog(subject, condition, channel_idx):
    # Get raw data
    raw = camcan_utils.get_raw(subject, condition)

    # Pick MEG magnetometers or gradiometers
    picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=True,
                           exclude='bads')
    picks_ch_names = [raw.ch_names[i] for i in picks]

    data = raw.get_data(picks)
    data = data[channel_idx, :]

    return data, raw


data_rest, _ = get_eog(subject, 'rest', 1)
data_task, _ = get_eog(subject, 'task', 1)


hurst_rest = mfa.compute_hurst(data_rest)
hurst_task = mfa.compute_hurst(data_task)


f = h5py.File('eog_rest_task.h5', 'w')
f.create_dataset('data_rest', data = data_rest)
f.create_dataset('data_task', data = data_task)
