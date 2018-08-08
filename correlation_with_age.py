import get_age
import numpy as np
import mf_results
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress

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
# Get subjects
subjects = mfr.mf_subjects

# Get age of subjects
ages = get_age.get_ages(subjects)


# Get average H over sensors (rest)
avg_H = mfr.all_log_cumulants_rest[:, :, 0].mean(axis = 1)


# Get average max_C2j over sensors (rest)
max_C2j = (mfr.all_cumulants_rest[:, :, 1, 9:14]).max(axis = 2)
avg_max_C2j = max_C2j.mean(axis = 1)


# Get averag  (max_C2j_rest - max_C2j_task) over sensors (rest)
diff_max_C2j = (mfr.all_cumulants_rest[:, :, 1, 9:14] - mfr.all_cumulants_task[:, :, 1, 9:14]).max(axis = 2)
avg_diff_max_C2j = max_C2j.mean(axis = 1)


# See correlation
quantity = avg_H
# quantity = avg_max_C2j
# quantity = avg_diff_max_C2j

corr, pval = pearsonr(ages, quantity)
print("Pearson correlation = %f, pvalue = %f"%(corr, pval))

corr2, pval2 = spearmanr(ages, quantity)
print("Spearman correlation = %f, pvalue = %f"%(corr2, pval2))


slope, intercept, r_value, p_value, std_err = linregress(ages,quantity)
x = np.array(ages)
y = slope*x+intercept

plt.figure()
plt.plot(ages, quantity, 'o')
plt.plot(x, y, 'r--', linewidth = 2)

plt.show()
