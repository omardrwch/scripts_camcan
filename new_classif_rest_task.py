import numpy as np
import mf_results
import matplotlib.pyplot as plt
import mne
import camcan_utils
import visualization_utils as v_utils
import os

import classification_utils as clf_utils
import h5py

RANDOM_STATE = 123
N_JOBS       = 1
np.random.seed(RANDOM_STATE)



PLOT_LEARNING    = True
PLOT_IMPORTANCES = True
SHOW_PLOTS       = True

SAVE             = False

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
mfr_eog = mf_results.get_results(params_index = params_index,
                                 sensor_type = 'EOG',
                                 conditions = ['rest', 'task'])


## --- Correcting the list of subjects
subjects_eog = mfr_eog.mf_subjects
subjects_mag = mfr.mf_subjects
subjects_both = list(set(subjects_eog).intersection(set(subjects_mag)))

subject_idx_eog = [ i for i in range(len(subjects_eog)) if subjects_eog[i] in  subjects_both]
subject_idx_mag = [ i for i in range(len(subjects_mag)) if subjects_mag[i] in  subjects_both]
##


#-------------------------------------------------------------------------------
# Global classification parameters
#-------------------------------------------------------------------------------

# Define cross validation scheme
n_splits   = 10
test_size  = 0.2          # used to obtain feature importances
scoring    = ['accuracy']


#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def get_features(features_choice, mfr):
    """
    Args:
        features_choice: 0 for c1 only
                         1 avg_C2(j) for 9<=j<=13
                         2 max C2(j) - min C2(j)
                         100 for c1 and c1 EOG
                         101 for avg_C2(j) and avg_C2(j) EOG

    Returns:
        X: matrix (2*n_subjects, n_features)
        y: array  (2*n_subjects,)
        subject_index: array (2*n_subjects)
    """
    X = []
    y = []
    subject_index = []

    n_subjects = mfr.n_subjects
    y_rest = np.zeros(n_subjects)
    y_task = np.ones(n_subjects)
    y      = np.hstack((y_rest, y_task))

    subject_index = np.hstack(( np.arange(n_subjects), np.arange(n_subjects)))


    if features_choice == 0:
        H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        H_task = mfr.all_log_cumulants_task[:, :, 0]

        X = np.vstack((H_rest, H_task))

    elif features_choice == 1000:
        c2_rest = mfr.all_log_cumulants_rest[:, :, 1].clip(max=0)  # (n_subjects, n_features)
        c2_task = mfr.all_log_cumulants_task[:, :, 1].clip(max=0)

        X = np.vstack((c2_rest, c2_task))


    elif features_choice == 1:
        avgC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).mean(axis = 2)  # (n_subjects, n_features)
        avgC2j_task = (mfr.all_cumulants_task[:, :, 1, 8:13]).mean(axis = 2)

        feat_rest = avgC2j_rest
        feat_task = avgC2j_task

        X      = np.vstack((feat_rest, feat_task))


    elif features_choice == 2:
        maxminC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).max(axis = 2) \
                         - (mfr.all_cumulants_rest[:, :, 1, 8:13]).min(axis = 2)# (n_subjects, n_features)
        maxminC2j_task = (mfr.all_cumulants_task[:, :, 1, 8:13]).max(axis = 2) \
                         - (mfr.all_cumulants_task[:, :, 1, 8:13]).min(axis = 2)

        feat_rest = maxminC2j_rest
        feat_task = maxminC2j_task

        X      = np.vstack((feat_rest, feat_task))

    elif features_choice == 3:
        maxC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).max(axis = 2)# (n_subjects, n_features)
        maxC2j_task = (mfr.all_cumulants_task[:, :, 1, 8:13]).max(axis = 2)

        feat_rest = maxC2j_rest
        feat_task = maxC2j_task

        X      = np.vstack((feat_rest, feat_task))


    elif features_choice == 100:
        H_rest_mag = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        H_task_mag = mfr.all_log_cumulants_task[:, :, 0]

        H_rest_eog = mfr_eog.all_log_cumulants_rest[subject_idx_eog, :, 0]  # (n_subjects, n_features)
        H_task_eog = mfr_eog.all_log_cumulants_task[subject_idx_eog, :, 0]

        H_rest = np.hstack((H_rest_mag, H_rest_eog))
        H_task = np.hstack((H_task_mag, H_task_eog))

        X      = np.vstack((H_rest, H_task))

    elif features_choice == 101:
        H_rest_eog = mfr_eog.all_log_cumulants_rest[subject_idx_eog, :, 0]  # (n_subjects, n_features)
        H_task_eog = mfr_eog.all_log_cumulants_task[subject_idx_eog, :, 0]

        H_rest = H_rest_eog
        H_task = H_task_eog

        X      = np.vstack((H_rest, H_task))

    elif features_choice == 200:
        maxminC2j_rest_mag = (mfr.all_cumulants_rest[:, :, 1, 8:13]).max(axis = 2) \
                            - (mfr.all_cumulants_rest[:, :, 1, 8:13]).min(axis = 2)# (n_subjects, n_features)
        maxminC2j_task_mag = (mfr.all_cumulants_task[:, :, 1, 8:13]).max(axis = 2) \
                            - (mfr.all_cumulants_task[:, :, 1, 8:13]).min(axis = 2)

        maxminC2j_rest_eog = (mfr_eog.all_cumulants_rest[subject_idx_eog, :, 1, 8:13]).max(axis = 2) \
                            - (mfr_eog.all_cumulants_rest[subject_idx_eog, :, 1, 8:13]).min(axis = 2)# (n_subjects, n_features)
        maxminC2j_task_eog = (mfr_eog.all_cumulants_task[subject_idx_eog, :, 1, 8:13]).max(axis = 2) \
                            - (mfr_eog.all_cumulants_task[subject_idx_eog, :, 1, 8:13]).min(axis = 2)


        feat_rest = np.hstack((maxminC2j_rest_mag, maxminC2j_rest_eog))
        feat_task = np.hstack((maxminC2j_task_mag, maxminC2j_task_eog))


        X      = np.vstack((feat_rest, feat_task))

    elif features_choice == 201:
        maxminC2j_rest_eog = (mfr_eog.all_cumulants_rest[subject_idx_eog, :, 1, 8:13]).max(axis = 2) \
                            - (mfr_eog.all_cumulants_rest[subject_idx_eog, :, 1, 8:13]).min(axis = 2)# (n_subjects, n_features)
        maxminC2j_task_eog = (mfr_eog.all_cumulants_task[subject_idx_eog, :, 1, 8:13]).max(axis = 2) \
                            - (mfr_eog.all_cumulants_task[subject_idx_eog, :, 1, 8:13]).min(axis = 2)


        feat_rest = maxminC2j_rest_eog
        feat_task = maxminC2j_task_eog


        X      = np.vstack((feat_rest, feat_task))

    elif features_choice == 300:
        H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        H_task = mfr.all_log_cumulants_task[:, :, 0]

        avgC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).mean(axis = 2)  # (n_subjects, n_features)
        avgC2j_task = (mfr.all_cumulants_task[:, :, 1, 8:13]).mean(axis = 2)


        feat_rest = np.hstack((H_rest, avgC2j_rest))   
        feat_task = np.hstack((H_task, avgC2j_task))   

        X      = np.vstack((feat_rest, feat_task))


    elif features_choice == 500:
        avgC2j_rest_eog = (mfr_eog.all_cumulants_rest[subject_idx_eog, :, 1, 8:13]).mean(axis = 2)  # (n_subjects, n_features)
        avgC2j_task_eog = (mfr_eog.all_cumulants_task[subject_idx_eog, :, 1, 8:13]).mean(axis = 2)

        feat_rest = avgC2j_rest_eog
        feat_task = avgC2j_task_eog

        X      = np.vstack((feat_rest, feat_task))

    elif features_choice == 501:

        avgC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).mean(axis = 2)  # (n_subjects, n_features)
        avgC2j_task = (mfr.all_cumulants_task[:, :, 1, 8:13]).mean(axis = 2)

        avgC2j_rest_eog = (mfr_eog.all_cumulants_rest[subject_idx_eog, :, 1, 8:13]).mean(axis = 2)  # (n_subjects, n_features)
        avgC2j_task_eog = (mfr_eog.all_cumulants_task[subject_idx_eog, :, 1, 8:13]).mean(axis = 2)

        feat_rest = np.hstack((avgC2j_rest, avgC2j_rest_eog)) 
        feat_task = np.hstack((avgC2j_task, avgC2j_task_eog)) 

        X      = np.vstack((feat_rest, feat_task))


    return X, y, subject_index


def plot_weights(weights, mfr, title = '', positive_only = False):
    # Load raw to get info about sensor positions
    raw_filename = 'sample_raw.fif'
    raw = mne.io.read_raw_fif(raw_filename)

    # raw = camcan_utils.get_raw(mfr.mf_subjects[0], 'rest')

    # get sensor positions via layout
    pos = mne.find_layout(raw.info).pos[mfr.channels_picks, :]

    # Plot
    if positive_only:
        vmax = np.max(weights)
        vmin = 0
        cmap = 'Reds'
    else:
        vmax = np.max(np.abs(weights))
        vmin = -vmax
        cmap = 'seismic'

    v_utils.plot_data_topo(weights, pos, vmin = vmin, vmax = vmax, title =title, cmap = cmap)


#-------------------------------------------------------------------------------
# Classification parameters 
#-------------------------------------------------------------------------------
# # Choose classifier
# classifier_name = 'random_forest_no_cv'

# # Choose features
# features = 0

for classifier_name in ['random_forest_no_cv']:
    for features in [1000]:

        features_str = None
        if features == 0:
            features_str = 'c1'
        elif features == 1:
            features_str = 'avgC2j'
        elif features == 2:
            features_str = 'maxminC2j'
        elif features == 3:
            features_str = 'maxC2j'
        elif features == 100:
            features_str = 'c1_EOGc1'
        elif features == 101:
            features_str = 'EOGc1'
        elif features == 200:
            features_str = 'maxminC2j_EOGmaxminC2j'
        elif features == 201:
            features_str = 'EOGmaxminC2j'
        elif features == 300:
            features_str = 'c1_avgC2j'
        elif features == 500:
            features_str = 'EOGavgC2j'
        elif features == 501:
            features_str = 'avgC2j_EOGavgC2j'
        elif features == 1000:
            features_str = 'c2'

        #===============================================================================
        # Load classification data
        #===============================================================================

        X, y, subject_index = get_features(features, mfr)

        #===============================================================================
        # Run classification
        #===============================================================================

        # List with sizes of training set
        train_sizes = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        # Get learning curve and feature importances
        train_sizes_abs, train_scores, test_scores, w, positive_only = \
            clf_utils.run_classification(   classifier_name, 
                                            X, y, subject_index,
                                            train_sizes,
                                            scoring,
                                            n_splits,
                                            RANDOM_STATE,
                                            N_JOBS, 
                                            ref_train_size = 1-test_size)


        if PLOT_LEARNING:
            clf_utils.plot_learning_curve(train_sizes_abs, train_scores, test_scores, title = classifier_name)

            if SAVE:
                # Save learning curve image 
                filename = '%s_%s_mf_%d_channel_%s.png'%(classifier_name, features_str, params_index, sensor_type)
                outdir   = os.path.join('outputs', 'learning_curves')
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                filename =  os.path.join(outdir, filename)
                plt.savefig(filename)
                del filename

                # Save learning curve raw data
                filename = '%s_%s_mf_%d_channel_%s.h5'%(classifier_name, features_str, params_index, sensor_type)
                outdir   = os.path.join('outputs', 'learning_curves_raw_data')
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                filename =  os.path.join(outdir, filename)   

                with h5py.File(filename, "w") as f:
                    f.create_dataset('train_sizes',  data = train_sizes_abs )
                    f.create_dataset('train_scores', data = train_scores )
                    f.create_dataset('test_scores',  data = test_scores )


            # Show curve
            if SHOW_PLOTS:
                plt.show()


        #===============================================================================
        # Plot feature importances
        #===============================================================================
        # try:
        if True:
            if PLOT_IMPORTANCES:
                plt.figure()
                plt.title('Feature importances')
                plt.plot(w, 'o'); 


                if len(w) <= mfr.n_channels + 2:
                    plot_weights(np.abs(w[0:mfr.n_channels]), mfr, positive_only = positive_only)
                else:
                    weights1 = w[0:mfr.n_channels]
                    weights2 = w[mfr.n_channels:2*mfr.n_channels]
                    plot_weights(weights1, mfr, positive_only = positive_only)
                    plot_weights(weights2, mfr, positive_only = positive_only)

                if SHOW_PLOTS:
                    plt.show()

                if SAVE:
                    filename = '%s_%s_mf_%d_channel_%s.png'%(classifier_name, features_str, params_index, sensor_type)
                    outdir   = os.path.join('outputs', 'feature_importances')
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    filename =  os.path.join(outdir, filename)
                    plt.savefig(filename)
                    del filename
        # except:
        #     pass


        del X
        del y
        plt.close()
