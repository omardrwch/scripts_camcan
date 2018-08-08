"""
Analyze correlation between (EOG decrease in self-similarity) and
(decrease of self-similarity of sensor i) for i in range(n_sensors)
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
from statsmodels.stats.multitest import multipletests
from scipy.stats import linregress, pearsonr, spearmanr



#-------------------------------------------------------------------------------
# Load results
#-------------------------------------------------------------------------------

# alpha for hyp testing
alpha = 0.01

# Choose index of MF parameters
params_index = 0

# Choose cumulant
cumulant_idx = 0

# EOG
mfr_eog = mf_results.get_results(params_index = params_index,
                             sensor_type =  'EOG',
                             conditions = ['rest', 'task'])
eog_logcumul_rest = mfr_eog.all_log_cumulants_rest[:,:, cumulant_idx] # shape (637, 2)
eog_logcumul_task = mfr_eog.all_log_cumulants_task[:,:, cumulant_idx]


# Magnetometers
mfr_mag = mf_results.get_results(params_index = params_index,
                                 sensor_type =  'mag',
                                 conditions = ['rest', 'task'])
mag_logcumul_rest = mfr_mag.all_log_cumulants_rest[:,:, cumulant_idx] # shape (627, 102)
mag_logcumul_task = mfr_mag.all_log_cumulants_task[:,:, cumulant_idx]


#-------------------------------------------------------------------------------
# 'Align' subjects: take all subject for whom we have all EOG results and all
# mag results
#-------------------------------------------------------------------------------
subjects_eog = mfr_eog.mf_subjects
subjects_mag = mfr_mag.mf_subjects
subjects_both = list(set(subjects_eog).intersection(set(subjects_mag)))

subject_idx_eog = [ i for i in range(len(subjects_eog)) if subjects_eog[i] in  subjects_both]
subject_idx_mag = [ i for i in range(len(subjects_mag)) if subjects_mag[i] in  subjects_both]


# shape (627, 2)
eog_logcumul_rest = eog_logcumul_rest[subject_idx_eog, :]
eog_logcumul_task = eog_logcumul_task[subject_idx_eog, :]

# shape (627, 102)
mag_logcumul_task = mag_logcumul_task[subject_idx_mag, :]
mag_logcumul_rest = mag_logcumul_rest[subject_idx_mag, :]


#-------------------------------------------------------------------------------
# Compute correlation between diferences
#-------------------------------------------------------------------------------
logcumul_diff_eog = eog_logcumul_rest - eog_logcumul_task # shape (627, 2)
logcumul_diff_mag = mag_logcumul_rest - mag_logcumul_task # shape (627, 102)

n_channels_eog = logcumul_diff_eog.shape[1]
n_channels_mag = logcumul_diff_mag.shape[1]


correlations = np.zeros((n_channels_eog, n_channels_mag))
pvalues      = np.zeros((n_channels_eog, n_channels_mag))


# Individual correlation
for eog_channel in range(n_channels_eog):
    diff_eog = logcumul_diff_eog[:, eog_channel]
    for mag_channel in range(n_channels_mag):
        diff_mag = logcumul_diff_mag[:, mag_channel]

        corr, pval = pearsonr(diff_eog, diff_mag)

        correlations[eog_channel, mag_channel] = corr
        pvalues[eog_channel, mag_channel]      = pval

# Apply FDR correction (separetely for each EOG channel)
pvalues[0, :] = multipletests(pvalues[0, :], alpha, method = 'fdr_bh')[1]
pvalues[1, :] = multipletests(pvalues[1, :], alpha, method = 'fdr_bh')[1]


# Set non significant correlations to zero
correlations[0, pvalues[0, :]>alpha] = 0.0
correlations[1, pvalues[1, :]>alpha] = 0.0


# Correlation between mean differences (average across channels)
eog_logcumul_diff_mean = logcumul_diff_eog.mean(axis = 1)
mag_logcumul_diff_mean = logcumul_diff_mag.mean(axis = 1)

corr_mean, pval_mean = pearsonr(eog_logcumul_diff_mean, mag_logcumul_diff_mean)
corr_mean_2, pval_mean_2 = spearmanr(eog_logcumul_diff_mean, mag_logcumul_diff_mean)


print("Correlation between \
(mean eog diff) and (mean cortex diff) = %0.5f, pvalue = %0.5f"%(corr_mean, pval_mean))

#-------------------------------------------------------------------------------
# Plot correlation
#-------------------------------------------------------------------------------

# Load raw to get info about sensor positions
raw = camcan_utils.get_raw(mfr_mag.mf_subjects[0], 'rest')
# get sensor positions via layout
pos = mne.find_layout(raw.info).pos[mfr_mag.channels_picks, :]
v_utils.plot_data_topo(correlations[0, :], pos, vmin = 0.0, vmax = 0.8, title = 'correlations for channel 1', cmap = 'Reds')
v_utils.plot_data_topo(correlations[1, :], pos, vmin = 0.0, vmax = 0.8, title = 'correlations for channel 2', cmap = 'Reds')

plt.show()

# if cumulant_idx == 0:
#   file_1 = os.path.join('output_images','correlation_eog_ch1_mf_%d.png'%params_index)
#   file_2 = os.path.join('output_images','correlation_eog_ch2_mf_%d.png'%params_index)
# elif cumulant_idx == 1:
#    file_1 = os.path.join('output_images','c2_correlation_eog_ch1_mf_%d.png'%params_index)
#    file_2 = os.path.join('output_images','c2_correlation_eog_ch2_mf_%d.png'%params_index)
