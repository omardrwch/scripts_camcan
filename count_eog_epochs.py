import numpy as np
import camcan_utils
import mfanalysis as mf
import matplotlib
import matplotlib.pyplot as plt
import os.path as op
import mne
import os
import h5py


# Subjects and conditions
subjects = camcan_utils.subjects[0:2]


n_eog_rest_list = np.zeros(len(subjects))
n_eog_task_list = np.zeros(len(subjects))


for ii, subject in enumerate(subjects):
    # Get raw data
    raw_rest = camcan_utils.get_raw(subject, 'rest')
    raw_task = camcan_utils.get_raw(subject, 'task')

    n_eog_epochs_rest = len(create_eog_epochs(raw_rest))
    n_eog_epochs_task = len(create_eog_epochs(raw_task))

    n_eog_rest_list[ii] = n_eog_epochs_rest
    n_eog_task_list[ii] = n_eog_epochs_task