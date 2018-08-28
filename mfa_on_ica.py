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

#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------

# Subjects and conditions
subjects = camcan_utils.subjects
conditions = camcan_utils.kinds

# MF parameters
params = mf_config.get_mf_params()[0]

# Subjects and conditions
subjects = camcan_utils.subjects
subject = subjects[5]
condition = 'task'

#-------------------------------------------------------------------------------
# Get data
#-------------------------------------------------------------------------------

# Get raw data
raw = camcan_utils.get_raw(subject, condition)

# Preprocess raw
raw, ica = camcan_utils.preprocess_raw(raw, subject, condition, n_components = 0.99)

# get sources
sources = ica.get_sources(raw)

# pick good ICA components
picks = mne.pick_types(sources.info, misc = True, eeg=False, stim=False, ecg=False, eog=False,exclude='bads')

# get data
data = sources.get_data(picks)


# #-------------------------------------------------------------------------------
# # MF analysis
# #-------------------------------------------------------------------------------
# mfa = mf.MFA(**params)
# mfa.verbose = 1
#
#
# N_COMPONENTS = 5 #data.shape[0]
# c2_list = []
# for ii in range(N_COMPONENTS):
#     signal = data[ii, :]
#     mfa.analyze(signal)
#     c2_list.append(mfa.cumulants.log_cumulants[1])
#     if ii == 0:
#         cumulants = mfa.cumulants
#     else:
#         cumulants.sum(mfa.cumulants)
# cumulants.plot()
# mfa.plt.show()
