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
import visualization_utils as v_utils


params_index = 0
cumul_idx    = 1

# Subjects and conditions
subjects = camcan_utils.subjects
conditions = camcan_utils.kinds

subject = subjects[11]

# Output folder
mf_io_info = mf_config.get_io_info()
camcan_output_dir = mf_io_info['camcan_output_dir'] + '_ica'

# file
subject_output_dir = os.path.join(camcan_output_dir, subject)
output_filename_rest = os.path.join(subject_output_dir, 'rest' + "_ica_params_%d"%(params_index) +'.h5')
output_filename_task = os.path.join(subject_output_dir, 'task' + "_ica_params_%d"%(params_index) +'.h5')


f_rest = h5py.File(output_filename_rest, 'r')
cumulants_rest = f_rest['cumulants'][:]
log_cumulants_rest = f_rest['log_cumulants'][:][:, cumul_idx]


f_task = h5py.File(output_filename_task, 'r')
cumulants_task = f_task['cumulants'][:]
log_cumulants_task = f_task['log_cumulants'][:][:, cumul_idx]


cumulants_list = [cumulants_rest[:, cumul_idx, :].mean(axis = 0), cumulants_task[:, cumul_idx, :].mean(axis = 0)]

v_utils.plot_cumulants(cumulants_list, j1=9, j2=13, title = '', labels = ['rest', 'task'], idx = cumul_idx)

print('rest: ',np.max(np.abs(log_cumulants_rest))- np.min(np.abs(log_cumulants_rest)))
print('task: ',np.max(np.abs(log_cumulants_task))- np.min(np.abs(log_cumulants_task)))


plt.figure()
plt.hist(log_cumulants_rest, bins = 20)
plt.hist(log_cumulants_task, bins = 20)



plt.show()
