"""
This script performs multifractal analysis on CamCAM data and generates
output files.
"""

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

# Channel type
MEG_TYPE = 'mag' # 'mag' or 'grad'

# Subjects and conditions
subjects = camcan_utils.subjects
conditions = camcan_utils.kinds

# MF parameters
param = {}
param['wt_name']   = 'db3'
param['formalism'] = None
param['p']         = np.inf
param['j1']        = 9
param['j2']        = 13
param['n_cumul']   = 3
param['gamint']    = 0.0
param['wtype']     = 0
mf_params = [param]

# Output folder
mf_io_info = mf_config.get_io_info()
camcan_output_dir = mf_io_info['camcan_output_dir'] + '_hurst'

if not os.path.exists(camcan_output_dir):
    os.makedirs(camcan_output_dir)

#-------------------------------------------------------------------------------
# MF analysis
#-------------------------------------------------------------------------------
def single_mf_analysis(args):
    """
    Apply MF analysis on (subject, condition) using the parameters in the
    mf_params dictionary
    """
    try:
        subject, condition, params_index_list, max_j = args

        # Get raw data
        raw = camcan_utils.get_raw(subject, condition)

        # Preprocess raw
        raw, ica = camcan_utils.preprocess_raw(raw, subject, condition)

        # Pick MEG magnetometers or gradiometers
        picks = mne.pick_types(raw.info, meg=MEG_TYPE, eeg=False, stim=False, eog=False,
                               exclude='bads')
        picks_ch_names = [raw.ch_names[i] for i in picks]

        data = raw.get_data(picks)

        # MF analysis object
        for params_index in params_index_list:
            params = mf_params[params_index]
            mfa = mf.MFA(**params)
            mfa.verbose = 1

            n_channels = len(picks)

            print("-------------------------------------------------")
            print("Performing mf analysis for (%s, %s)"%(subject, condition))
            print("-------------------------------------------------")
            all_hurst = np.zeros(n_channels)
            all_log2_Sj_2 = np.zeros((n_channels, max_j))

            for ii in range(n_channels):
                # Run analysis
                signal = data[ii, :]
                hurst = mfa.compute_hurst(signal)
                log2_Sj_2 = mfa.hurst_structure

                all_hurst[ii] = hurst
                all_log2_Sj_2[ii, :max_j] = log2_Sj_2[:max_j]

            # Save data
            subject_output_dir = os.path.join(camcan_output_dir, subject)
            if not os.path.exists(subject_output_dir):
                os.makedirs(subject_output_dir)

            output_filename = os.path.join(subject_output_dir, condition + "_channel_%s_hurst"%(MEG_TYPE) +'.h5')

            with h5py.File(output_filename, "w") as f:
                params_string = np.string_(str(params))
                f.create_dataset('params', data = params_string )
                f.create_dataset('hurst', data = all_hurst )
                f.create_dataset('log2_Sj_2', data = all_log2_Sj_2 )
                f.create_dataset('channels_picks', data = picks)
                channels_name_list = [n.encode("ascii", "ignore") for n in picks_ch_names]
                f.create_dataset('picks_ch_names', (len(channels_name_list),1),'S10', channels_name_list)

            print("-------------------------------------------------")
            print("*** saved file ", output_filename)
            print("-------------------------------------------------")

    except:
        pass
    # return raw, output_filename


if __name__ == '__main__':
    params_index_list = [0]
    # Select params
    max_j = 15

    arg_instances = []
    for ss in subjects:
        for cond in ['rest', 'task']:
                arg_instances.append( (ss, cond, params_index_list, max_j) )


    # remove already computed instances
    new_arg_instances = []
    for args in arg_instances:
        subject, condition, params_index_list, max_j = args
        subject_output_dir = os.path.join(camcan_output_dir, subject)
        output_filename = os.path.join(subject_output_dir, condition + "_channel_%s_hurst"%(MEG_TYPE) +'.h5')
        if op.isfile(output_filename):
            continue
        else:
            new_arg_instances.append(args)
    arg_instances = new_arg_instances

    Parallel(n_jobs=4, verbose=1, backend="multiprocessing")(map(delayed(single_mf_analysis), arg_instances))
