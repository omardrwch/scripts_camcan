"""
Analyze results in data_mf_out/camcan/EOG. The results are organized by mf_results.py
"""

import numpy as np
import camcan_utils
import mfanalysis as mf
import matplotlib
import matplotlib.pyplot as plt
import os.path as op
import mne
import os
import h5py
import mf_results
import visualization_utils as v_utils

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, cross_validate


#-------------------------------------------------------------------------------
# Load results
#-------------------------------------------------------------------------------

# Choose index of MF parameters
params_index = 0

# Choose sensor type
sensor_type = 'EOG'
mfr = mf_results.get_results(params_index = params_index,
                             sensor_type = sensor_type,
                             conditions = ['rest', 'task'])


# H_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).max(axis = 2)  #mfr.all_log_cumulants_rest[:,:, 0]
# H_task = (mfr.all_cumulants_task[:, :, 1, 8:13]).max(axis = 2)  #mfr.all_log_cumulants_task[:,:, 0]
H_rest = mfr.all_log_cumulants_rest[:,:, 0]
H_task = mfr.all_log_cumulants_task[:,:, 0]


#-------------------------------------------------------------------------------
# Plots
#-------------------------------------------------------------------------------

plt.figure()
plt.title('H - EOG channel 1')
plt.plot(H_rest[:, 0], 'bo-', label='rest')
plt.plot(H_task[:, 0], 'ro-', label='task')
plt.xlabel('subject')
plt.ylabel('H')
plt.legend()
plt.grid()


plt.figure()
plt.title('H - EOG channel 2')
plt.plot(H_rest[:, 1], 'bo-', label='rest')
plt.plot(H_task[:, 1], 'ro-', label='task')
plt.xlabel('subject')
plt.ylabel('H')
plt.legend()
plt.grid()


plt.figure()
plt.title('H both channels')
plt.plot(H_rest[:, 0], H_rest[:, 1], 'bo', label='rest')
plt.plot(H_task[:, 0], H_task[:, 1], 'ro', label='task')
plt.xlabel('H - EOG channel 1')
plt.ylabel('H - EOG channel 2')
# Find NaNs
subjects = np.arange(mfr.n_subjects)

bad = np.logical_or( np.isnan(H_rest), np.isnan(H_task) )
bad_subject = np.logical_or(bad[:,0], bad[:,1])

H_rest = H_rest[~bad_subject, :]
H_task = H_task[~bad_subject, :]
subjects = subjects[~bad_subject]


C1j_rest_ch1 = (mfr.all_cumulants_rest[~bad_subject, 0, 0, :]).mean(axis = 0)
C1j_rest_ch2 = (mfr.all_cumulants_rest[~bad_subject, 1, 0, :]).mean(axis = 0)

C1j_task_ch1 = (mfr.all_cumulants_task[~bad_subject, 0, 0, :]).mean(axis = 0)
C1j_task_ch2 = (mfr.all_cumulants_task[~bad_subject, 1, 0, :]).mean(axis = 0)



C2j_rest_ch1 = (mfr.all_cumulants_rest[~bad_subject, 0, 1, :]).mean(axis = 0)
C2j_rest_ch2 = (mfr.all_cumulants_rest[~bad_subject, 1, 1, :]).mean(axis = 0)

C2j_task_ch1 = (mfr.all_cumulants_task[~bad_subject, 0, 1, :]).mean(axis = 0)
C2j_task_ch2 = (mfr.all_cumulants_task[~bad_subject, 1, 1, :]).mean(axis = 0)



v_utils.plot_cumulants([C1j_rest_ch1, C1j_task_ch1],
                        j1=9, j2=13,
                        title = 'C1(j) EOG channel 1',
                        labels = ['rest', 'task'])

v_utils.plot_cumulants([C1j_rest_ch2, C1j_task_ch2],
                        j1=9, j2=13,
                        title = 'C1(j) EOG channel 2',
                        labels = ['rest', 'task'])


v_utils.plot_cumulants([C2j_rest_ch1, C2j_task_ch1],
                        j1=9, j2=13,
                        title = 'C2(j) EOG channel 1',
                        labels = ['rest', 'task'])

v_utils.plot_cumulants([C2j_rest_ch2, C2j_task_ch2],
                        j1=9, j2=13,
                        title = 'C2(j) EOG channel 2',
                        labels = ['rest', 'task'])



#-------------------------------------------------------------------------------
# Classification
#-------------------------------------------------------------------------------
subject_index = np.hstack((subjects, subjects))
X = np.vstack((H_rest,H_task))
y0 = np.zeros(H_rest.shape[0])
y1 = np.ones(H_task.shape[0])
y  = np.hstack((y0,y1))

cv  = GroupShuffleSplit(n_splits= 50,
                        test_size = 0.25,
                        random_state = 123 )

svm = SVC(kernel='linear')
# # parameters for grid search
# p_grid = {}
# p_grid['C'] = np.power(10.0, np.linspace(-4, 4, 10))
# # classifier
# clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=cv)
# # parameters required to fit the classifier
# fit_params = {'groups':subject_index}
clf = svm
fit_params = {}

output = cross_validate(clf, X = X, y = y, scoring = ['accuracy'], cv = cv,
                        groups = subject_index, return_train_score = True,
                        fit_params=fit_params, verbose = 2,
                        n_jobs = 6)

print("Train accuracy = %0.4f +- %0.4f"%(output['train_accuracy'].mean(), output['train_accuracy'].std()))
print("Test accuracy = %0.4f +- %0.4f"%(output['test_accuracy'].mean(), output['test_accuracy'].std()))


svm.fit(X, y)
print("weights = ", svm.coef_)

plt.show()
