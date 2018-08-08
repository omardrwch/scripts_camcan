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
MEG_TYPE = 'grad' # 'mag' or 'grad'

# Subjects and conditions
subjects = camcan_utils.subjects
conditions = camcan_utils.kinds

# MF parameters
mf_params = mf_config.get_mf_params()

# Output folder
mf_io_info = mf_config.get_io_info()
camcan_output_dir = mf_io_info['camcan_output_dir']

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
            all_log_cumulants = np.zeros((n_channels, mfa.n_cumul))
            all_cumulants     = np.zeros((n_channels, mfa.n_cumul, max_j))
            all_hmin          = np.zeros(n_channels)

            for ii in range(n_channels):
                # Run analysis
                signal = data[ii, :]
                mfa.analyze(signal)
                log_cumulants = mfa.cumulants.log_cumulants # shape (3,)
                cumulants     = mfa.cumulants.values  # shape (3, 16)
                hmin          = mfa.hmin

                all_log_cumulants[ii, :] = log_cumulants
                all_cumulants[ii, :, :max_j] = cumulants[:, :max_j]
                all_hmin[ii]  = hmin

            # Save data
            subject_output_dir = os.path.join(camcan_output_dir, subject)
            if not os.path.exists(subject_output_dir):
                os.makedirs(subject_output_dir)

            output_filename = os.path.join(subject_output_dir, condition + "_channel_%s_params_%d"%(MEG_TYPE, params_index) +'.h5')

            with h5py.File(output_filename, "w") as f:
                params_string = np.string_(str(params))
                f.create_dataset('params', data = params_string )
                f.create_dataset('log_cumulants', data = all_log_cumulants )
                f.create_dataset('cumulants', data = all_cumulants )
                f.create_dataset('hmin', data = all_hmin )
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
    params_index_list = [0, 1, 2, 3]
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
        output_filename = os.path.join(subject_output_dir, condition + "_channel_%s_params_%d"%(MEG_TYPE, params_index_list[-1]) +'.h5')
        if op.isfile(output_filename):
            continue
        else:
            new_arg_instances.append(args)
    arg_instances = new_arg_instances

    Parallel(n_jobs=1, verbose=1, backend="threading")(map(delayed(single_mf_analysis), arg_instances))
