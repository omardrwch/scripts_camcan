import numpy as np
import mf_results
import matplotlib.pyplot as plt
import mne
import camcan_utils
import visualization_utils as v_utils


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, cross_validate, GridSearchCV
from sklearn.svm import LinearSVC


N_JOBS = 1

# Set seed
SEED      = 123
np.random.seed(SEED)


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

#-------------------------------------------------------------------------------
# Classification parameters
#-------------------------------------------------------------------------------
classifier_choice = 0
features_choice   = 0

# Number of GroupShuffleSplit splits and test size
n_splits = 50
test_size = 0.95


#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def get_features(features_choice, mfr):
    """
    Args:
        features_choice: 0 for H only
                         1 for H and max_C2(j) for 9<=j<=13
                         2 for max_C2(j) for 9<=j<=13 only
                         100 for H only, INCLUDING EOG

    Returns:
        X: matrix (2*n_subjects, n_features)
        y: array  (2*n_subjects,)
        subject_index: array (2*n_subjects)
    """
    X = []
    y = []
    subject_index = []
    if features_choice == 0:
        H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        H_task = mfr.all_log_cumulants_task[:, :, 0]
        n_subjects = mfr.n_subjects
        y_rest = np.zeros(n_subjects)
        y_task = np.ones(n_subjects)

        X      = np.vstack((H_rest, H_task))
        y      = np.hstack((y_rest, y_task))
        subject_index = np.hstack(( np.arange(n_subjects), np.arange(n_subjects)))


    elif features_choice == 1:
        H_rest = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        H_task = mfr.all_log_cumulants_task[:, :, 0]
        maxC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).max(axis = 2)  # (n_subjects, n_features)
        maxC2j_task = (mfr.all_cumulants_task[:, :, 1, 8:13]).max(axis = 2)

        feat_rest = np.hstack((H_rest, maxC2j_rest))
        feat_task = np.hstack((H_task, maxC2j_task))

        n_subjects = mfr.n_subjects
        y_rest = np.zeros(n_subjects)
        y_task = np.ones(n_subjects)

        X      = np.vstack((feat_rest, feat_task))
        y      = np.hstack((y_rest, y_task))
        subject_index = np.hstack(( np.arange(n_subjects), np.arange(n_subjects)))



    elif features_choice == 2:
        maxC2j_rest = (mfr.all_cumulants_rest[:, :, 1, 8:13]).max(axis = 2)  # (n_subjects, n_features)
        maxC2j_task = (mfr.all_cumulants_task[:, :, 1, 8:13]).max(axis = 2)

        n_subjects = mfr.n_subjects
        y_rest = np.zeros(n_subjects)
        y_task = np.ones(n_subjects)

        X      = np.vstack((maxC2j_rest, maxC2j_task))
        y      = np.hstack((y_rest, y_task))
        subject_index = np.hstack(( np.arange(n_subjects), np.arange(n_subjects)))



    elif features_choice == 100:
        H_rest_mag = mfr.all_log_cumulants_rest[:, :, 0]  # (n_subjects, n_features)
        H_task_mag = mfr.all_log_cumulants_task[:, :, 0]

        ## ---
        subjects_eog = mfr_eog.mf_subjects
        subjects_mag = mfr.mf_subjects
        subjects_both = list(set(subjects_eog).intersection(set(subjects_mag)))

        subject_idx_eog = [ i for i in range(len(subjects_eog)) if subjects_eog[i] in  subjects_both]
        subject_idx_mag = [ i for i in range(len(subjects_mag)) if subjects_mag[i] in  subjects_both]
        ##

        H_rest_eog = mfr_eog.all_log_cumulants_rest[subject_idx_eog, :, 0]  # (n_subjects, n_features)
        H_task_eog = mfr_eog.all_log_cumulants_task[subject_idx_eog, :, 0]

        H_rest = np.hstack((H_rest_mag, H_rest_eog))
        H_task = np.hstack((H_task_mag, H_task_eog))

        n_subjects = mfr.n_subjects
        y_rest = np.zeros(n_subjects)
        y_task = np.ones(n_subjects)

        X      = np.vstack((H_rest, H_task))
        y      = np.hstack((y_rest, y_task))
        subject_index = np.hstack(( np.arange(n_subjects), np.arange(n_subjects)))



    return X, y, subject_index

def run_classification(classifier_choice, X, y, subject_index):
    """
    Args:
        classifier_choice: 0 for random forest
    """
    gkf = GroupShuffleSplit(n_splits= n_splits, test_size = test_size)

    if classifier_choice == 0:
        clf = RandomForestClassifier(n_estimators = 300,
                                     random_state=SEED)
        fit_params = {}

    elif classifier_choice == 1:
        svm = LinearSVC(random_state=SEED, dual = False)
        # parameters for grid search
        p_grid = {}
        p_grid['C'] = np.power(10.0, np.linspace(-4, 4, 10))
        # classifier
        clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=gkf)
        fit_params = {'groups':subject_index}

    elif classifier_choice == 2:
        clf = RandomForestClassifier(n_estimators = 300,
                                     random_state=SEED,
                                     max_features = 1)
        fit_params = {}                                 


    clf.fit(X, y, **fit_params)
    y_pred = clf.predict(X)




    scoring = ['accuracy']
    scores = cross_validate(clf,
                            X, y,
                            scoring = scoring,
                            cv = gkf,
                            groups = subject_index,
                            return_train_score = True,
                            fit_params=fit_params,
                            verbose = 2,
                            n_jobs = N_JOBS)

    test_accs  = scores['test_accuracy']
    train_accs = scores['train_accuracy']

    print("")
    print("* train acc (%): ")
    print("-- mean = ", 100*train_accs.mean(), ", std = ", 100*train_accs.std())
    print("* test acc:(%) ")
    print("-- mean = ", 100*test_accs.mean(), ", std = ", 100*test_accs.std())
    print("* number of folds: ", n_splits)




    try:
        w = clf.best_estimator_.coef_.squeeze()
    except:
        pass

    try:
        w = clf.feature_importances_
    except:
        pass

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

    v_utils.plot_data_topo(weights, pos, vmin = vmin, vmax = vmax, title =title, cmap = 'Reds')


#-------------------------------------------------------------------------------
# Run
#-------------------------------------------------------------------------------
X, y, subject_index = get_features(features_choice, mfr)

output = run_classification(classifier_choice, X, y, subject_index)


plt.figure()
plt.title('Feature importances')
plt.plot(output['w'], 'o'); plt.show()


if len(output['w']) <= mfr.n_channels + 2:
    plot_weights(np.abs(output['w'][0:mfr.n_channels]), mfr)
    plt.show()
else:
    weights1 = output['w'][0:mfr.n_channels]
    weights2 = output['w'][mfr.n_channels:2*mfr.n_channels]
    plot_weights(np.abs(weights1), mfr)
    plot_weights(np.abs(weights2), mfr)
    plt.show()



# gkf = output['cross_val']
# for train_index, test_index in gkf.split(X, y, groups = subject_index):
#     print("TRAIN:", subject_index[train_index], "TEST:", subject_index[test_index])
#     print("TRAIN:", len(train_index), "TEST:", len(test_index))
#     set1 = set(subject_index[train_index])
#     set2 = set(subject_index[test_index])
#     print("!!!!! ", set1.intersection(set2))
#     print("     ")
