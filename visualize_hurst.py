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
import hurst_results
import classification_utils as clf_utils
from scipy.stats import ttest_rel
from scipy.stats import pearsonr, spearmanr

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20


# Load results
sensor_type = 'mag'
hurst_data = hurst_results.get_results(sensor_type=sensor_type)


#-------------------------------------------------------------------------------
# Averages and topomaps
#-------------------------------------------------------------------------------

# Load raw to get info about sensor positions
# raw_filename = 'sample_raw.fif'
# raw          = mne.io.read_raw_fif(raw_filename)
raw = camcan_utils.get_raw(hurst_data.mf_subjects[0], 'rest')

# get sensor positions via layout
pos = mne.find_layout(raw.info).pos[hurst_data.channels_picks, :]

vmax = np.abs(hurst_data.all_hurst_rest.mean(axis=0)).max()
vmin = -vmax
v_utils.plot_data_topo(hurst_data.all_hurst_rest.mean(axis=0), pos, vmin = vmin, vmax = vmax, title = 'H rest', cmap = 'seismic')
v_utils.plot_data_topo(hurst_data.all_hurst_task.mean(axis=0), pos, vmin = vmin, vmax = vmax, title = 'H task', cmap = 'seismic')
vmax = np.abs((hurst_data.all_hurst_rest - hurst_data.all_hurst_task).mean(axis=0)).max()
vmin = 0
v_utils.plot_data_topo(hurst_data.all_hurst_rest.mean(axis=0) - hurst_data.all_hurst_task.mean(axis=0),
                        pos, vmin = vmin, vmax = vmax, title = '(H rest) - (H task)', cmap = 'Reds')


#-------------------------------------------------------------------------------
# Average log2(S(j, 2))
#-------------------------------------------------------------------------------
def visualize_structure(sensor_name):
    sensor_index = hurst_data.ch_name2index[sensor_name]

    log2Sj2_rest = hurst_data.all_log2_Sj_2_rest[:, sensor_index, :]
    log2Sj2_task = hurst_data.all_log2_Sj_2_task[:, sensor_index, :]

    v_utils.plot_cumulants_2( [log2Sj2_rest, log2Sj2_task ],
                    title ='Mean structure function - ' + sensor_name,
                    labels = ['rest', 'task'], idx = '$\log_2(S(j,2))$')

visualize_structure('MEG0311')
visualize_structure('MEG1841')

raw.plot_sensors()
plt.show()
