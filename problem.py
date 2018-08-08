import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
import mne


# Subjects
with open('subjects_maxfilter.txt', 'r') as fid:
    subjects = [ll.strip('\n') for ll in fid.readlines()]


def get_raw(subject, kind):
    # Get raw
    fname = op.join(
            cfg.camcan_meg_raw_path,
            subject, kind, '%s_raw.fif' % kind)
    raw = mne.io.read_raw_fif(fname)
    return raw

years = []
problem = []
ids = []
for subject in subjects:
    try:
        raw = get_raw(subject, 'rest')
        birthyear = raw.info['subject_info']['birthday'][0]
        ids.append(raw.info['subject_info']['id'])
        years.append(birthyear)
    except:
        problem.append(subject)

years = np.array(years)
ok_years  = np.logical_and(years > 1900, years < 2018)  # remove some weird values like 5874777 or 5874531
years = years[ok_years]

age = 2018 - years

plt.hist(age)
plt.show()
