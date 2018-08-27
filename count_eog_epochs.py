import numpy as np
import camcan_utils
import mfanalysis as mf
import matplotlib
import matplotlib.pyplot as plt
import os.path as op
import mne
import os
import h5py
from statsmodels import robust


from mne.preprocessing import create_eog_epochs, create_ecg_epochs


# Subjects and conditions
subjects = camcan_utils.subjects#[0:10]


# n_eog_rest_list = []#np.zeros(len(subjects))
# n_eog_task_list = []#np.zeros(len(subjects))
#
#
# for ii, subject in enumerate(subjects):
#     # Get raw data
#     try:
#         raw_rest = camcan_utils.get_raw(subject, 'rest')
#         raw_task = camcan_utils.get_raw(subject, 'task')
#
#         n_eog_epochs_rest = len(create_eog_epochs(raw_rest))
#         n_eog_epochs_task = len(create_eog_epochs(raw_task))
#
#         n_eog_rest_list.append(n_eog_epochs_rest)
#         n_eog_task_list.append(n_eog_epochs_task)
#     except:
#         continue
#
# n_eog_rest_list = np.array(n_eog_rest_list)
# n_eog_task_list = np.array(n_eog_task_list)
#
# with h5py.File("n_eog.h5", "w") as f:
#     f.create_dataset("n_eog_rest_list", data=n_eog_rest_list)
#     f.create_dataset("n_eog_task_list", data=n_eog_task_list)

f = h5py.File("n_eog.h5", "r")
n_eog_rest_list = f['n_eog_rest_list'][:]
n_eog_task_list = f['n_eog_task_list'][:]

print('rest: %0.5f +- %0.5f'%(n_eog_rest_list.mean(), n_eog_rest_list.std()))
print('task: %0.5f +- %0.5f'%(n_eog_task_list.mean(), n_eog_task_list.std()))
print(" ")
print('median/mad rest: %0.5f / %0.5f'%(np.median(n_eog_rest_list), robust.mad(n_eog_rest_list)))
print('median/mad task: %0.5f / %0.5f'%(np.median(n_eog_task_list), robust.mad(n_eog_task_list)))
