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
import mf_results_ica 
import classification_utils as clf_utils
from scipy.stats import ttest_rel

from scipy.stats import pearsonr, spearmanr


from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20


# MF parameters
params_index = 2

# Feature choice for log-cumulants
feature = None

# Maximum number of ICA components to consider
max_components = None # None to use all components

# Load results
mfr = mf_results_ica.get_results(params_index = params_index, 
                                 conditions = ['rest', 'task'], 
                                 feature = feature,
                                 max_components = max_components)

# Classification parameters
train_size = 0.75
classifier_name = 'linear_svm'
groups = np.arange(mfr.n_subjects)
groups = np.hstack((groups, groups))
y      = np.hstack(( np.zeros(mfr.n_subjects), np.ones(mfr.n_subjects) ))

#---------------------------------------------------------------------------------
# Visualize mean cumulants
#---------------------------------------------------------------------------------
# avg_cumulants_rest = mfr.cumulants_rest[:,:,:].mean(axis = 0)
# avg_cumulants_task = mfr.cumulants_task[:,:,:].mean(axis = 0)

v_utils.plot_cumulants_2([mfr.cumulants_rest[:,0,:], mfr.cumulants_task[:,0,:]], j1=9, j2=13, 
                        title = '$C_1^\mathrm{avg}(j)$', labels = ['rest', 'task'], idx = 0)
v_utils.plot_cumulants_2([mfr.cumulants_rest[:,1,:], mfr.cumulants_task[:,1,:]], j1=9, j2=13, 
                        title = '$C_2^\mathrm{avg}(j)$', labels = ['rest', 'task'], idx = 1)



def run_vis_and_classif(feats_rest, feats_task, title = '', feat_name = '', feat_dim = 1):

    if feat_dim == 1:
        print("--")
        plt.figure()
        plt.hist(feats_rest, bins = 30, label='rest')
        plt.hist(feats_task, bins = 30, label='task', alpha = 0.8)
        plt.title(title)
        plt.legend()


        t, pval = ttest_rel(feats_rest, feats_task)
        print(feat_name," pval = ", pval)

    X = np.vstack(( feats_rest.reshape(-1, feat_dim), feats_task.reshape(-1, feat_dim)   ))
    scores = clf_utils.simple_classification(classifier_name, 
                                             X, y, groups, 
                                             train_size)

    print("%s crossval score = %0.3f +- %0.3f"%(feat_name, scores.mean(), scores.std()))

# #---------------------------------------------------------------------------------
# # Vis and classif - Mean c1
# #---------------------------------------------------------------------------------
# run_vis_and_classif(mfr.mean_c1_rest, mfr.mean_c1_task, title = 'mean $c_1$', feat_name = 'mean c1', feat_dim = 1)


# #---------------------------------------------------------------------------------
# # Vis and classif - Max c1
# #---------------------------------------------------------------------------------
# run_vis_and_classif(mfr.max_c1_rest, mfr.max_c1_task, title = 'max $c_1$', feat_name = 'max c1', feat_dim = 1)


# #---------------------------------------------------------------------------------
# # Vis and classif - Min c1
# #---------------------------------------------------------------------------------
# run_vis_and_classif(mfr.min_c1_rest, mfr.min_c1_task, title = 'min $c_1$', feat_name = 'min c1', feat_dim = 1)



#---------------------------------------------------------------------------------
# Check correlation with EOG
#---------------------------------------------------------------------------------
import mf_results
mfr_eog = mf_results.get_results(params_index = params_index,
                             sensor_type =  'EOG',
                             conditions = ['rest', 'task'])

subjects_eog = mfr_eog.mf_subjects
subjects_mag = mfr.mf_subjects
subjects_both = list(set(subjects_eog).intersection(set(subjects_mag)))
subject_idx_eog = [ i for i in range(len(subjects_eog)) if subjects_eog[i] in  subjects_both]


eog_logcumul_rest_ch1 = mfr_eog.all_log_cumulants_rest[subject_idx_eog,0, 0] # shape (637, 2)
eog_logcumul_task_ch1 = mfr_eog.all_log_cumulants_task[subject_idx_eog,0, 0]
eog_logcumul_rest_ch2 = mfr_eog.all_log_cumulants_rest[subject_idx_eog,1, 0] # shape (637, 2)
eog_logcumul_task_ch2 = mfr_eog.all_log_cumulants_task[subject_idx_eog,1, 0]




feats_eog_rest = mfr_eog.all_log_cumulants_rest[subject_idx_eog,:, 0]
feats_eog_task = mfr_eog.all_log_cumulants_task[subject_idx_eog,:, 0]

feats_eog_maxc1_rest = np.hstack(( feats_eog_rest, mfr.max_c1_rest.reshape(-1,1)  ))
feats_eog_maxc1_task = np.hstack(( feats_eog_task, mfr.max_c1_task.reshape(-1,1)  ))


run_vis_and_classif(feats_eog_rest, feats_eog_task, title = '$c_1^\mathrm{EOG}$', feat_name = 'c1 EOG', feat_dim = 2)
run_vis_and_classif(feats_eog_maxc1_rest, feats_eog_maxc1_task, title = '$c_1^\mathrm{max}$ and $c_1^\mathrm{EOG}$', feat_name = 'c1max and c1 EOG', feat_dim = 3)


corr_1, pval_1 = pearsonr(eog_logcumul_task_ch1 - eog_logcumul_rest_ch1, mfr.max_c1_task - mfr.max_c1_rest)
corr_2, pval_2 = pearsonr(eog_logcumul_task_ch2 - eog_logcumul_rest_ch2, mfr.max_c1_task - mfr.max_c1_rest)

# corr_1, pval_1 = spearmanr(eog_logcumul_task_ch1 - eog_logcumul_rest_ch1, mfr.max_c1_task - mfr.max_c1_rest)
# corr_2, pval_2 = spearmanr(eog_logcumul_task_ch2 - eog_logcumul_rest_ch2, mfr.max_c1_task - mfr.max_c1_rest)

print("Correlation between max_c1 and EOG_ch1 = %f, pvalue %f"%(corr_1, pval_1))
print("Correlation between max_c1 and EOG_ch2 = %f, pvalue %f"%(corr_2, pval_2))


plt.show()



