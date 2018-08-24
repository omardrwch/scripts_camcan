"""
Perfom hypothesis testin on results in data_mf_out/camcan.
The results are organized by mf_results.py
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

from statsmodels.stats.multitest import multipletests
import visualization_utils as v_utils
import mf_results

matplotlib.rcParams.update({'errorbar.capsize': 2})


#-------------------------------------------------------------------------------
# Load results
#-------------------------------------------------------------------------------

# Choose index of MF parameters
params_index = 0

# Choose sensor type
sensor_type = 'mag'
mfr = mf_results.get_results(params_index = params_index,
                             sensor_type = sensor_type,
                             conditions = ['rest', 'task'])


#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------
# Load raw to get info about sensor positions
raw_filename = 'sample_raw.fif'
raw = mne.io.read_raw_fif(raw_filename)
# raw = camcan_utils.get_raw(mfr.mf_subjects[0], 'rest')

# get sensor positions via layout
pos = mne.find_layout(raw.info).pos[mfr.channels_picks, :]


correction_multiple_tests = 'fdr' # 'fdr', 'bonferroni' or None
alpha = 0.005

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def one_sided_ttest_rel(x,y):
    """
    Returns the value of the statistic and p-value for the paired Student t test
    with the hypotheses:
        H_0:  x = y
        H_1:  x < y

    For wilcoxon test, see::
        https://stackoverflow.com/questions/16296225/one-sided-wilcoxon-signed-rank-test-using-scipy
    """
    stat, pval = ttest_rel(x,y) # two-sided p-value, we need the one-sided!
    T = stat

    if T > 0: # x > y
        pval = 1.0-pval/2.0
    else:
        pval = pval/2.0

    return stat, pval

def two_sided_ttest_rel(x,y):
    """
    Returns the value of the statistic and p-value for the paired Student t test
    with the hypotheses:
        H_0:  x = y
        H_1:  x != y
    """
    stat, pval = ttest_rel(x,y) # two-sided p-value, we need the one-sided!
    T = stat

    return stat, pval

def pvals_correction(pvals):
    if correction_multiple_tests == 'fdr':
        # - Benjamini/Hochberg  (non-negative) =  'indep' in mne.fdr_correction
        _, pvals, _, _ = multipletests(pvals, alpha, method = 'fdr_bh')

    elif correction_multiple_tests == 'bonferroni':
                # bonferroni
        _, pvals, _, _ = multipletests(pvals, alpha, method = 'bonferroni')

    return pvals


#-------------------------------------------------------------------------------
# Test whether H_task - H_rest < 0 for each sensor
# Hypotheses:
# H_0:  H_task - H_rest = 0
# H_1:  H_task - H_rest != 0
#-------------------------------------------------------------------------------

H_rest = mfr.all_log_cumulants_rest[:, :, 0] # shape (n_subjects, n_sensors)
H_task = mfr.all_log_cumulants_task[:, :, 0] # shape (n_subjects, n_sensors)

H_pvals = np.ones(mfr.n_channels)

for ii in range(mfr.n_channels):
    H_rest_ii = H_rest[:, ii]
    H_task_ii = H_task[:, ii]
    stat, pval = two_sided_ttest_rel(H_task_ii, H_rest_ii)

    H_pvals[ii] = pval


# correction for multiple comparisons
H_pvals = pvals_correction(H_pvals)
H_signif = H_pvals < alpha


# Plot significant differences:
H_diff = -(H_task.mean(axis=0) - H_rest.mean(axis=0))
H_diff[~H_signif] = 0.0
v_utils.plot_data_topo(H_diff, pos, title = '(H_rest - H_task) tested for H_task != H_rest', cmap = 'Reds')

#-------------------------------------------------------------------------------
# Test whether max(C2(j)_task) - max(C2(j)_rest) != 0 for each sensor
#-------------------------------------------------------------------------------

# compute max of C2(j)
max_C2j_rest = mfr.all_cumulants_rest[:,:,:,8:13].max(axis = 3)
max_C2j_rest = max_C2j_rest[:,:,1]   # shape (n_subjects, n_sensors)
max_C2j_task = mfr.all_cumulants_task[:,:,:,8:13].max(axis = 3)
max_C2j_task = max_C2j_task[:,:,1]   # shape (n_subjects, n_sensors)


maxC2j_pvals = np.ones(mfr.n_channels)

for ii in range(mfr.n_channels):
    maxC2j_rest_ii = max_C2j_rest[:, ii]
    maxC2j_task_ii = max_C2j_task[:, ii]
    stat, pval = two_sided_ttest_rel(maxC2j_task_ii, maxC2j_rest_ii)

    maxC2j_pvals[ii] = pval


# correction for multiple comparisons
maxC2j_pvals = pvals_correction(maxC2j_pvals)
maxC2j_signif = maxC2j_pvals < alpha


# Plot significant differences:
avgC2j_diff = -(max_C2j_task.mean(axis=0) - max_C2j_rest.mean(axis=0))
avgC2j_diff[~maxC2j_signif] = 0.0
v_utils.plot_data_topo(avgC2j_diff, pos, title = '(max_C2j_rest - max_C2j_task) tested for max_C2j_task != max_C2j_rest', cmap = 'Reds')


plt.show()
