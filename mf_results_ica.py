"""
This scripts reads the results in data_mf_out/camcan and organize them
for further analysis.

IMPORTANT:
 The following parameters must be set:
    - conditions (default = ['rest', 'task'])
    - params_index (0, 1, 2 or 3)
    - sensor_type ('mag', 'grad' or 'EOG')


Available data:
    - unexisting_files (list): files that were not generated by the code that performs
                               MF analysis, due to some exception (I noticed exceptions
                               happen during ICA sometimes)
    - subjects_incomplete (list): subjects for which at least one file is missing

    - mf_subjects (list): all subjects that are not in subjects_incomplete, that is
                          the ones who have all MF files available

    - n_subjects  = len(mf_subjects)

    - cumulants_rest: average of C_i(j) for all ICA components in rest, shape (n_subjects, n_cumulants, nj)

    - cumulants_task: average of C_i(j) for all ICA components in task, shape (n_subjects, n_cumulants, nj)


"""

import mf_config
import camcan_utils
import h5py
import numpy as np
import os.path as op

class MF_Results():
    """
    Stores data described above
    """
    def __init__(self):
        self.params = None
        self.sensor_type = None
        self.unexisting_files = None
        self.subjects_incomplete = None
        self.mf_subjects = None
        self.n_subjects = None

def get_results(params_index = 0, conditions = ['rest', 'task'], feature = None, max_components = None, ):

    #-------------------------------------------------------------------------------
    # Parameters
    #-------------------------------------------------------------------------------
    # Subjects and conditions
    subjects = camcan_utils.subjects
    # conditions = ['rest', 'task'] #camcan_utils.kinds

    # MF parameters
    mf_params = mf_config.get_mf_params()

    # Output folder
    mf_io_info = mf_config.get_io_info()
    camcan_output_dir = mf_io_info['camcan_output_dir']
    camcan_output_dir += '_ica'

    print(camcan_output_dir)
    #-------------------------------------------------------------------------------
    # Read files
    #-------------------------------------------------------------------------------

    # debug
    unexisting_files = []
    subjects_incomplete = [] # subjects with incomplete mf analysis
    n_channels_list = []
    #

    # dictionary to store results
    mf_results = {}

    for subject in subjects:
        mf_results[subject] = {}
        for condition in conditions:
            filename = op.join(camcan_output_dir,
                               subject,
                               condition + "_ica_params_%d"%params_index +'.h5')


            if op.isfile(filename):
                mf_results[subject][condition] = {}
                with h5py.File(filename, "r") as f:
                    log_cumulants = f['log_cumulants'][:]
                    cumulants = f['cumulants'][:]
                    hmin = f['hmin'][:]
                    params = f['params'].value
                    picks_ch_names = [name.decode('ascii') for name in f['picks_ch_names'][:].squeeze()]
                    channels_picks = f['channels_picks'][:]
                    n_channels_list.append(len(picks_ch_names))

                    # store results in dictionary
                    mf_results[subject][condition]['log_cumulants'] = log_cumulants
                    mf_results[subject][condition]['cumulants']     = cumulants
                    mf_results[subject][condition]['hmin']          = hmin
                    mf_results[subject][condition]['channels_picks']= channels_picks
                    mf_results[subject][condition]['picks_ch_names']= picks_ch_names
            else:
                unexisting_files.append(filename)
                subjects_incomplete.append(subject)

    subjects_incomplete = list(set(subjects_incomplete))


    #-------------------------------------------------------------------------------
    # Organize data in arrays (easier to do averages etc)
    #-------------------------------------------------------------------------------

    # Subjects with all mf analysis computed
    mf_subjects = [s for s in subjects if s not in subjects_incomplete]
    n_subjects  = len(mf_subjects)

    cumulants_rest = np.zeros((n_subjects, 3, 15))
    cumulants_task = np.zeros((n_subjects, 3, 15))

    # c1 features  ~~~~ 
    mean_c1_rest   = np.zeros(n_subjects)
    mean_c1_task   = np.zeros(n_subjects)

    max_c1_rest   = np.zeros(n_subjects)
    max_c1_task   = np.zeros(n_subjects)

    min_c1_rest   = np.zeros(n_subjects)
    min_c1_task   = np.zeros(n_subjects)
    # ~~~~ 


    for subject_idx, subject in enumerate(mf_subjects):
        for condition in conditions:
            cumulants     = mf_results[subject][condition]['cumulants']
            log_cumulants = mf_results[subject][condition]['log_cumulants']
            hmin          = mf_results[subject][condition]['hmin']

            if max_components is not None:
                cumulants = cumulants[:max_components, :, :]
                log_cumulants = log_cumulants[:max_components, :]
                hmin      = hmin[:max_components]


            if condition == 'rest':
                cumulants_rest[subject_idx, :, :] = cumulants.mean(axis = 0)

                # c1 features  ~~~~ 
                mean_c1_rest[subject_idx] = log_cumulants[:, 0].mean(axis = 0)
                max_c1_rest[subject_idx] = log_cumulants[:, 0].max(axis = 0)
                min_c1_rest[subject_idx] = log_cumulants[:, 0].min(axis = 0)
                # ~~~~ 

            if condition == 'task':
                cumulants_task[subject_idx, :, :] = cumulants.mean(axis = 0)

                # c1 features  ~~~~ 
                mean_c1_task[subject_idx] = log_cumulants[:, 0].mean(axis = 0)
                max_c1_task[subject_idx] = log_cumulants[:, 0].max(axis = 0)
                min_c1_task[subject_idx] = log_cumulants[:, 0].min(axis = 0)
                # ~~~~ 

        # print(log_cumulants.shape)

    #-------------------------------------------------------------------------------
    # Store and return
    #-------------------------------------------------------------------------------
    results = MF_Results()
    results.params      = mf_params[params_index]
    results.unexisting_files = unexisting_files
    results.subjects_incomplete = subjects_incomplete
    results.mf_subjects = mf_subjects
    results.n_subjects = n_subjects
    results.cumulants_rest = cumulants_rest
    results.cumulants_task = cumulants_task


    results.mean_c1_rest = mean_c1_rest
    results.mean_c1_task = mean_c1_task

    results.max_c1_rest  = max_c1_rest
    results.max_c1_task  = max_c1_task

    results.min_c1_rest  = min_c1_rest
    results.min_c1_task  = min_c1_task


    return results
