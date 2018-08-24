"""
Classification into groups of ages.
"""


import numpy as np
import mf_results
import matplotlib.pyplot as plt
import mne
import camcan_utils
import visualization_utils as v_utils
import get_age


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate

# Set seed
SEED      = 123
np.random.seed(SEED)


#-------------------------------------------------------------------------------
# Load results
#-------------------------------------------------------------------------------

# Choose index of MF parameters
params_index = 2

# Choose sensor type
sensor_type = 'mag'
mfr = mf_results.get_results(params_index = params_index,
                             sensor_type = sensor_type,
                             conditions = ['rest', 'task'])


#-------------------------------------------------------------------------------
# Classification parameters
#-------------------------------------------------------------------------------
classifier_choice = 0
features_choice   = 1

# Number of sujects in the test set
split_factor = 50


#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def get_features(features_choice, mfr, age_group = 'two_groups'):
    """
    Args:
        features_choice: 0 for H resting state
                         1 for H and max_C2(j) for 9<=j<=13 resting state
                         2 for max_C2(j) for 9<=j<=13 only resting state
                         3 for H task
                         4 for H and max_C2(j) for 9<=j<=13 task
                         5 for max_C2(j) for 9<=j<=13 only task
                         6 for H and max_C2(j) for 9<=j<=13  rest and task
                         7 for H and C2(j) for 9<=j<=13 resting state
    Returns:
        X: matrix (n_subjects, n_features)
        y: array  (n_subjects,)
        subject_index: array (n_subjects)
    """
    X = []
    if age_group == 'two_groups':
        ages = np.array(get_age.get_ages(mfr.mf_subjects))
        y    = ages >= 54.0  # mean age ~= median age = 54

    subject_index = np.arange(mfr.n_subjects)


    if features_choice == 0:
        H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        n_subjects = mfr.n_subjects
        X  = H_rest

    elif features_choice == 1:
        H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        maxC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 9:14]).max(axis = 2)  # (n_subjects, n_features)
        X  = np.hstack((H_rest, maxC2j_rest))


    elif features_choice == 2:
        maxC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 9:14]).max(axis = 2)  # (n_subjects, n_features)
        X      = maxC2j_rest

    elif features_choice == 3:
        H_task = mfr.all_log_cumulants_task[:, :, 0]  # (n_subjects, n_features)
        n_subjects = mfr.n_subjects
        X  = H_task

    elif features_choice == 4:
        H_task = mfr.all_log_cumulants_task[:, :, 0]  # (n_subjects, n_features)
        maxC2j_task = (mfr.all_cumulants_task[:, :, 1, 9:14]).max(axis = 2)  # (n_subjects, n_features)
        X  = np.hstack((H_task, maxC2j_task))

    elif features_choice == 5:
        maxC2j_task = (mfr.all_cumulants_task[:, :, 1, 9:14]).max(axis = 2)  # (n_subjects, n_features)
        X      = maxC2j_task


    elif features_choice == 6:
        subject_index = np.hstack((subject_index,subject_index))
        y             = np.hstack((y, y))
        H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        maxC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 9:14]).max(axis = 2)  # (n_subjects, n_features)
        H_task = mfr.all_log_cumulants_task[:, :, 0]  # (n_subjects, n_features)
        maxC2j_task = (mfr.all_cumulants_task[:, :, 1, 9:14]).max(axis = 2)  # (n_subjects, n_features)

        X_rest  = np.hstack((H_rest, maxC2j_rest))
        X_task  = np.hstack((H_task, maxC2j_task))
        X       = np.vstack( (X_rest, X_task) )  # add more examples, not more features


    # elif features_choice == 7:
    #     H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
    #     C2j_rest = (mfr.all_cumulants_rest[:, :, 1, 9:14]).max(axis = 2)  # (n_subjects, n_features)
    #



    return X, y, subject_index

def run_classification(classifier_choice, X, y, subject_index):
    """
    Args:
        classifier_choice: 0 for random forest
    """
    n_subjects = X.shape[0]//2

    if classifier_choice == 0:
        clf = RandomForestClassifier(n_estimators = 300,
                                     random_state=SEED)

    n_splits = n_subjects // split_factor

    clf.fit(X, y)
    y_pred = clf.predict(X)

    gkf = GroupKFold(n_splits= n_splits)

    scoring = ['accuracy']
    scores = cross_validate(clf,
                            X, y,
                            scoring = scoring,
                            cv = gkf,
                            groups = subject_index,
                            return_train_score = True)

    test_accs  = scores['test_accuracy']
    train_accs = scores['train_accuracy']

    print("")
    print("* train acc (%): ")
    print("-- mean = ", 100*train_accs.mean(), ", std = ", 100*train_accs.std())
    print("* test acc:(%) ")
    print("-- mean = ", 100*test_accs.mean(), ", std = ", 100*test_accs.std())
    print("* number of folds: ", n_splits)


    # Get weights
    w = clf.feature_importances_

    output = {}
    output['test_accs']      = test_accs
    output['test_accs_mean'] = test_accs.mean()
    output['test_accs_std']  = test_accs.std()
    output['cross_val']      = gkf
    output['w']              = w

    return output


def plot_weights(weights, mfr, title = '', positive_only = True):
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
    else:
        vmax = np.max(np.abs(weights))
        vmin = -vmax

    v_utils.plot_data_topo(weights, pos, vmin = vmin, vmax = vmax, title =title)


#-------------------------------------------------------------------------------
# Run
#-------------------------------------------------------------------------------
X, y, subject_index = get_features(features_choice, mfr)

output = run_classification(classifier_choice, X, y, subject_index)

if len(output['w']) == mfr.n_channels:
    plot_weights(output['w'], mfr)
    plt.show()
else:
    try:
        weights1 = output['w'][:mfr.n_channels]
        weights2 = output['w'][mfr.n_channels:]
        plot_weights(weights1, mfr)
        plot_weights(weights2, mfr)
        plt.show()
    except:
        pass



# gkf = output['cross_val']
# for train_index, test_index in gkf.split(X, y, groups = subject_index):
#     print("TRAIN:", subject_index[train_index], "TEST:", subject_index[test_index])
#     print("TRAIN:", len(train_index), "TEST:", len(test_index))
#     set1 = set(subject_index[train_index])
#     set2 = set(subject_index[test_index])
#     print("!!!!! ", set1.intersection(set2))
#     print("     ")
