"""
Use regression tree to predict age.
"""

import numpy as np
import mf_results
import matplotlib.pyplot as plt
import mne
import camcan_utils
import visualization_utils as v_utils
import get_age



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_squared_error,median_absolute_error ,make_scorer

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
                             sensor_type  = sensor_type,
                             conditions   = ['rest', 'task'])


#-------------------------------------------------------------------------------
# Regression parameters
#-------------------------------------------------------------------------------
estimator_choice  = 0
features_choice   = 1

# Number of sujects in the test set
split_factor = 50

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def get_features(features_choice, mfr):
    """
    Args:
        features_choice: 0 for H resting state
                         1 for H and max_C2(j) for 9<=j<=13 resting state
                         2 for max_C2(j) for 9<=j<=13 only resting state

    Returns:
        X: matrix (n_subjects, n_features)
        y: array  (n_subjects,)
        subject_index: array (n_subjects)
    """
    X = []
    y = np.array(get_age.get_ages(mfr.mf_subjects))
    subject_index = np.arange(mfr.n_subjects)

    if features_choice == 0:
        H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        n_subjects = mfr.n_subjects

        X  = H_rest

    elif features_choice == 1:
        H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        maxC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).max(axis = 2)  # (n_subjects, n_features)
        X  = np.hstack((H_rest, maxC2j_rest))


    elif features_choice == 2:
        maxC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).max(axis = 2)  # (n_subjects, n_features)
        X      = maxC2j_rest

    return X, y, subject_index


def plot_weights(weights, mfr, title = '', positive_only = True):
    # Load raw to get info about sensor positions
    raw = camcan_utils.get_raw(mfr.mf_subjects[0], 'rest')

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


def run_classification(estimator_choice, X, y, subject_index):
    """
    Args:
        estimator_choice: 0 for random forest
    """
    n_subjects = X.shape[0]//2

    if estimator_choice == 0:
        estimator = RandomForestRegressor(n_estimators = 300,
                                          random_state=SEED)

    n_splits = n_subjects // split_factor

    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    gkf = GroupKFold(n_splits= n_splits)


    scorer = make_scorer(median_absolute_error, greater_is_better = False)
    # scorer = make_scorer(mean_squared_error, greater_is_better = False)
    scores = cross_validate(estimator,
                            X, y,
                            cv = gkf,
                            groups = subject_index,
                            return_train_score = True) # scoring = scorer,

    # test_mse  = -scores['test_score']
    # train_mse  = -scores['train_score']
    # sqrt_test_mse  = np.sqrt(test_mse)
    # sqrt_train_mse = np.sqrt(train_mse)


    train_median_abs_err  = -scores['train_score']
    test_median_abs_err  = -scores['test_score']


    print("")
    print("* train median abs error: ")
    print("-- mean = ", 1*train_median_abs_err.mean(), ", std = ", 1*train_median_abs_err.std())
    print("* test median abs error: ")
    print("-- mean = ", 1*test_median_abs_err.mean(), ", std = ", 1*test_median_abs_err.std())
    print("* number of folds: ", n_splits)


    # Get weights
    w = estimator.feature_importances_

    output = {}
    output['scores'] = scores
    output['train_median_abs_err'] = train_median_abs_err
    output['test_median_abs_err'] = test_median_abs_err
    output['cross_val']      = gkf
    output['w']              = w

    return output
#-------------------------------------------------------------------------------
# Run
#-------------------------------------------------------------------------------
X, y, subject_index = get_features(features_choice, mfr)

output = run_classification(estimator_choice, X, y, subject_index)

if len(output['w']) == mfr.n_channels:
    plot_weights(output['w'], mfr)
    plt.show()
else:
    weights1 = output['w'][:mfr.n_channels]
    weights2 = output['w'][mfr.n_channels:]
    plot_weights(weights1, mfr)
    plot_weights(weights2, mfr)
    plt.show()
