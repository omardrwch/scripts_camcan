"""
Analyze results in data_mf_out/camcan. The results are organized by mf_results.py
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

matplotlib.rcParams.update({'errorbar.capsize': 2})


def get_scales(fs, min_f, max_f):
    """
    Compute scales corresponding to the analyzed frequencies
    """
    f0 = (3.0/4.0)*fs
    j1 = int(np.ceil(np.log2(f0/max_f)))
    j2 = int(np.ceil(np.log2(f0/min_f)))
    return j1, j2

def get_freqs(fs, j1, j2):
    f0 = (3.0/4.0)*fs
    f1 = np.power(2.0, -j2)*f0
    f2 = np.power(2.0, -j1)*f0

    return f1, f2

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
# Averages and topomaps
#-------------------------------------------------------------------------------

# Load raw to get info about sensor positions
raw = camcan_utils.get_raw(mfr.mf_subjects[0], 'rest')

# get sensor positions via layout
pos = mne.find_layout(raw.info).pos[mfr.channels_picks, :]

# compute averages across subjects
avg_log_cumulants_rest = mfr.all_log_cumulants_rest.mean(axis = 0)
avg_log_cumulants_task = mfr.all_log_cumulants_task.mean(axis = 0)

# compute stds across subjects
std_log_cumulants_rest = mfr.all_log_cumulants_rest.std(axis = 0)
std_log_cumulants_task = mfr.all_log_cumulants_task.std(axis = 0)

# Plot
vmin = np.min(avg_log_cumulants_task[:, 0])
vmax = np.max(avg_log_cumulants_task[:, 0])

v_utils.plot_data_topo(avg_log_cumulants_rest[:, 0], pos, vmin = vmin, vmax = vmax, title = 'H rest')
v_utils.plot_data_topo(avg_log_cumulants_task[:, 0], pos, vmin = vmin, vmax = vmax, title = 'H task')
# plot_data_topo(std_log_cumulants_rest[:, 0], pos, title = 'std H rest')
# plot_data_topo(std_log_cumulants_task[:, 0], pos, title = 'std H task')


vmin = -np.min(avg_log_cumulants_task[:, 1])
vmax = -np.max(avg_log_cumulants_task[:, 1])

v_utils.plot_data_topo(-1*avg_log_cumulants_rest[:, 1].clip(max = 0), pos, vmin = vmin, vmax = vmax, title = 'M rest')
v_utils.plot_data_topo(-1*avg_log_cumulants_task[:, 1].clip(max = 0), pos, vmin = vmin, vmax = vmax, title = 'M task')

v_utils.plot_data_topo(avg_log_cumulants_rest[:, 0] - avg_log_cumulants_task[:, 0], pos, title = 'H rest - task', cmap = 'Reds')
# plot_data_topo(avg_log_cumulants_task[:, 1].clip(max = 0) - avg_log_cumulants_rest[:, 1].clip(max = 0), pos, title = 'M rest - task')


#-------------------------------------------------------------------------------
# C1(j) and C2(j)
#-------------------------------------------------------------------------------

# compute averages across subjects
avg_cumulants_rest = mfr.all_cumulants_rest.mean(axis = 0) # shape (n_channels, 3,15)
avg_cumulants_task = mfr.all_cumulants_task.mean(axis = 0)


# compare maximum value of C_2(j) over [j1=9, j2=13]
mfr.all_cumulants_rest[:,:,:, 8:13].max()

max_cumulants_rest_9_13 = (mfr.all_cumulants_rest[:,:,:, 8:13].max(axis = 3)).mean(axis = 0)  #(avg_cumulants_rest[:,:,8:13]).mean(axis = 2)
max_cumulants_task_9_13 = (mfr.all_cumulants_task[:,:,:, 8:13].max(axis = 3)).mean(axis = 0)

vmin = np.min(max_cumulants_task_9_13[:, 1])
vmax = np.max(max_cumulants_task_9_13[:, 1])
v_utils.plot_data_topo(max_cumulants_rest_9_13[:, 1], pos, vmin = vmin, vmax = vmax, title = 'Max C_2(j) rest')
v_utils.plot_data_topo(max_cumulants_task_9_13[:, 1], pos, vmin = vmin, vmax = vmax, title = 'Max C_2(j) task')
v_utils.plot_data_topo(max_cumulants_rest_9_13[:, 1]-max_cumulants_task_9_13[:, 1], pos, title = 'Max C_2(j) rest-task', cmap = 'Reds')


if mfr.sensor_type == 'mag':
    # Sensor for rest/task comparison
    sensor1_name = 'MEG0511'
    sensor1_index = mfr.ch_name2index[sensor1_name]

    v_utils.plot_cumulants( [avg_cumulants_rest[sensor1_index, 0, :], avg_cumulants_task[sensor1_index, 0, :] ],
                    title ='H rest/task - ' + sensor1_name,
                    labels = ['rest', 'task'])

    # Sensors for comparison of different regions (rest)
    sensor2_name = 'MEG0811'
    sensor3_name = 'MEG1841'
    sensor2_index = mfr.ch_name2index[sensor2_name]
    sensor3_index = mfr.ch_name2index[sensor3_name]

    v_utils.plot_cumulants( [avg_cumulants_rest[sensor2_index, 0, :], avg_cumulants_rest[sensor3_index, 0, :] ],
                    title ='H rest - ' + sensor2_name + ' vs. ' + sensor3_name,
                    labels = [sensor2_name, sensor3_name])


    # Sensor compare M rest vs task
    sensor4_name = 'MEG1621'
    sensor4_index = mfr.ch_name2index[sensor4_name]
    v_utils.plot_cumulants( [avg_cumulants_rest[sensor4_index, 1, :], avg_cumulants_task[sensor4_index, 1, :] ],
                    title ='M rest/task - ' + sensor4_name,
                    labels = ['rest', 'task'])

    sensor5_name = 'MEG0811'
    sensor5_index = mfr.ch_name2index[sensor5_name]
    v_utils.plot_cumulants( [avg_cumulants_rest[sensor5_index, 1, :], avg_cumulants_task[sensor5_index, 1, :] ],
                    title ='M rest/task - ' + sensor5_name,
                    labels = ['rest', 'task'])


    v_utils.plot_sensors([sensor1_name], pos, mfr)
    v_utils.plot_sensors([sensor2_name, sensor3_name], pos, mfr)
    v_utils.plot_sensors([sensor4_name], pos, mfr)


#-------------------------------------------------------------------------------
# C1(j) and C2(j) for all sensors
#-------------------------------------------------------------------------------
avg_all_sensors_cumulants_rest = avg_cumulants_rest.mean(axis = 0) # shape (3,15)
avg_all_sensors_cumulants_task = avg_cumulants_task.mean(axis = 0)
std_all_sensors_cumulants_rest = avg_cumulants_rest.std(axis = 0)  # shape (3,15)
std_all_sensors_cumulants_task = avg_cumulants_task.std(axis = 0)


plt.figure()
plt.title('C_1(j) - average over all sensors')
plt.errorbar(np.arange(1, 16), avg_all_sensors_cumulants_rest[0, :],fmt ='bo--', label = 'rest', yerr = std_all_sensors_cumulants_rest[0, :])
plt.errorbar(np.arange(1, 16), avg_all_sensors_cumulants_task[0, :], fmt ='ro--', label = 'task', yerr = std_all_sensors_cumulants_task[0, :])
plt.xlabel('j')
plt.grid()
plt.legend()


plt.figure()
plt.title('C_2(j) - average over all sensors')
plt.errorbar(np.arange(1, 16), avg_all_sensors_cumulants_rest[1, :],fmt ='bo--', label = 'rest', yerr = std_all_sensors_cumulants_rest[1, :])
plt.errorbar(np.arange(1, 16), avg_all_sensors_cumulants_task[1, :], fmt ='ro--', label = 'task', yerr = std_all_sensors_cumulants_task[1, :])
plt.xlabel('j')
plt.grid()
plt.legend()






raw.plot_sensors()
plt.show()
