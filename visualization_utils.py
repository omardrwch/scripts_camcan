"""
Plots of cumulants, topomaps etc.
"""
import mne
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress


def plot_data_topo(data, pos, vmin = np.min, vmax= np.max,  title = '', cmap = 'viridis'):
    # plt.figure()
    fig, ax_topo = plt.subplots(1, 1, figsize=(7, 5))
    image, _ = mne.viz.plot_topomap(data, pos,
                         vmin = vmin, vmax = vmax, show=False, cmap = cmap)

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(title)


def plot_cumulants(cumulants_list, j1=9, j2=13, title = '', labels = None, idx = None):
    colors = ['b', 'r']
    x_reg  = np.arange(j1, j2+1)
    plt.figure()
    plt.title(title)
    for ii, cumulants in enumerate(cumulants_list):
        x_plot = np.arange(1, len(cumulants)+1)
        y_plot = cumulants
        y_reg  = cumulants[j1-1:j2]
        if labels is not None:
            plt.plot(x_plot, y_plot, colors[ii]+'o--', alpha = 0.75)
        else:
            plt.plot(x_plot, y_plot, colors[ii]+'o--', alpha = 0.75)

        # linear regression
        log2_e  = np.log2(np.exp(1))
        slope, intercept, r_value, p_value, std_err = linregress(x_reg,y_reg)
        y1 = slope*j1 + intercept
        y2 = slope*j2 + intercept
        log_cumul = log2_e*slope
        plt.plot( [j1, j2], [y1, y2], colors[ii]+'-', linewidth=2,
                  label = labels[ii]+', slope*log2(e) = %0.3f'%log_cumul)


    plt.xlabel('j')
    if idx is None:
        plt.ylabel('C(j)')
    else:
        plt.ylabel('$C_%d(j)$'%(idx+1))
    plt.legend()
    plt.grid()

def plot_cumulants_2(cumulants_list, j1=9, j2=13, title = '', labels = None, idx = None):
    colors = ['b', 'r']
    x_reg  = np.arange(j1, j2+1)
    plt.figure()
    plt.title(title)
    for ii, cumulants in enumerate(cumulants_list):
        x_plot = np.arange(1, cumulants.shape[1]+1)
        y_plot = cumulants.mean(axis = 0)
        y_std  = cumulants.std(axis = 0)

        y_reg  = y_plot[j1-1:j2]
        if labels is not None:
            plt.plot(x_plot, y_plot, colors[ii]+'o--', alpha = 0.75)
        else:
            plt.plot(x_plot, y_plot, colors[ii]+'o--', alpha = 0.75)

        plt.fill_between(x_plot, y_plot - y_std,
                         y_plot + y_std, alpha=0.25,
                         color=colors[ii])

        # linear regression
        log2_e  = np.log2(np.exp(1))
        slope, intercept, r_value, p_value, std_err = linregress(x_reg,y_reg)
        y1 = slope*j1 + intercept
        y2 = slope*j2 + intercept
        log_cumul = log2_e*slope
        plt.plot( [j1, j2], [y1, y2], colors[ii]+'-', linewidth=2,
                  label = labels[ii]+', slope*log2(e) = %0.3f'%log_cumul)


    plt.xlabel('j')
    if idx in [0, 1]:
        plt.ylabel('$C_%d(j)$'%(idx+1))
    else:
        plt.ylabel(idx)

    plt.legend()
    plt.grid()


def plot_sensors(sensor_name_list, pos, mfr):
    for sensor_name in sensor_name_list:
        sensor_index = mfr.ch_name2index[sensor_name]
        foo_data_sensor = np.zeros(mfr.n_channels)
        foo_data_sensor[sensor_index] = 1
        plot_data_topo(foo_data_sensor, pos, title = sensor_name)
