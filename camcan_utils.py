import os.path as op
from collections import Counter

import numpy as np
import pandas as pd
import mne

from joblib import Parallel, delayed
from autoreject import get_rejection_threshold

import config as cfg
import library as lib
import matplotlib.pyplot as plt

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs


#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------
SEED = 123

# Subjects
with open('subjects_maxfilter.txt', 'r') as fid:
    subjects = [ll.strip('\n') for ll in fid.readlines()]


# subjects2 = lib.utils.get_subjects(cfg.camcan_meg_raw_path)

# Condition
kinds = ['passive', 'rest','task']

max_filter_info_path = op.join(
    cfg.camcan_meg_path,
    "data_nomovecomp/"
    "aamod_meg_maxfilt_00001")

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def _parse_bads(subject, kind):
    sss_log = op.join(
        max_filter_info_path, subject,
        kind, "mf2pt2_{kind}_raw.log".format(kind=kind))

    try:
        bads = lib.preprocessing.parse_bad_channels(sss_log)
    except Exception as err:
        print(err)
        bads = []
    # first 100 channels ommit the 0.
    bads = [''.join(['MEG', '0', bb.split('MEG')[-1]])
            if len(bb) < 7 else bb for bb in bads]
    return bads

def _run_maxfilter(raw, subject, kind):
    bads = _parse_bads(subject, kind)
    raw.info['bads'] = bads
    raw = lib.preprocessing.run_maxfilter(raw, coord_frame='head')
    return raw


def _compute_add_ssp_exg(raw):
    reject_eog, reject_ecg = _get_global_reject_ssp(raw)

    proj_eog = mne.preprocessing.compute_proj_eog(
        raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1)

    proj_ecg = mne.preprocessing.compute_proj_ecg(
        raw, average=True, reject=reject_ecg, n_mag=1, n_grad=1, n_eeg=1)

    raw.add_proj(proj_eog[0])
    raw.add_proj(proj_ecg[0])

def _get_global_reject_ssp(raw):
    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    if len(eog_epochs) >= 5:
        reject_eog = get_rejection_threshold(eog_epochs, decim=8)
        del reject_eog['eog']
    else:
        reject_eog = None

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    if len(ecg_epochs) >= 5:
        reject_ecg = get_rejection_threshold(ecg_epochs[:200], decim=8)
    else:
        reject_eog = None

    if reject_eog is None:
        reject_eog = reject_ecg
    if reject_ecg is None:
        reject_ecg = reject_eog
    return reject_eog, reject_ecg

def get_bad_magnetometers(kind_id = 1):
    all_bads = []
    for id in range(len(subjects)):
        subject = subjects[id]
        kind    = kinds[kind_id]

        bads = _parse_bads(subject, kind)
        all_bads += bads

        print(bads)
    all_bads = list(set(all_bads))

    # Select subject and condition
    subject = subjects[1]
    kind    = kinds[0]

    # Get raw
    fname = op.join(
            cfg.camcan_meg_raw_path,
            subject, kind, '%s_raw.fif' % kind)
    raw = mne.io.read_raw_fif(fname)

    # Pick MEG magnetometers
    picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False,
                           exclude='bads')
    # Get magnetometers names
    picks_ch_names = [raw.ch_names[i] for i in picks]

    # Find bad magnetometers
    bad_mag = list(set(picks_ch_names) & set(all_bads))

    return bad_mag

def get_bad_magnetometers_after_maxfilter(kind_id = 1, step = 10):
    all_bads = []
    for id in range(0,len(subjects), step):
        try:
            subject = subjects[id]
            kind    = kinds[kind_id]

            # Get raw
            fname = op.join(
                    cfg.camcan_meg_raw_path,
                    subject, kind, '%s_raw.fif' % kind)
            raw = mne.io.read_raw_fif(fname)

            mne.channels.fix_mag_coil_types(raw.info)

            # Apply maxfilter
            raw = _run_maxfilter(raw, subject, kind)

            bads = raw.info['bads']
            all_bads += bads

            print(bads)
        except:
            continue

    all_bads = list(set(all_bads))

    # Select subject and condition
    subject = subjects[1]
    kind    = kinds[0]

    # Get raw
    fname = op.join(
            cfg.camcan_meg_raw_path,
            subject, kind, '%s_raw.fif' % kind)
    raw = mne.io.read_raw_fif(fname)

    # Pick MEG magnetometers
    picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False,
                           exclude='bads')
    # Get magnetometers names
    picks_ch_names = [raw.ch_names[i] for i in picks]

    # Find bad magnetometers
    bad_mag = list(set(picks_ch_names) & set(all_bads))

    return bad_mag


def get_ica(raw, method = 'fastica',
            n_components = 0.99,
            bad_seconds = None,
            decim = 10,
            n_max_ecg = 3,
            n_max_eog = 2,
            max_iter  = 250,
            reject = dict(mag=5e-12, grad=4000e-13),
            random_state = SEED,
            plot = False):
    """
    Fit ICA to raw.

    Args:
        raw
        n_components: see mne.preprocessing.ICA
        fmin        : cutoff frequency of the high-pass filter before ICA
        bad_seconds : the first 'bad_seconds' seconds of data is annotated as 'BAD'
                      and not used for ICA. If None, no annotation is done
        decim       : see mne.preprocessing.ICA.fit()
        n_max_ecg   : maximum number of ECG components to remove
        n_max_eog   : maximum number of EOG components to remove
        max_iter    : maximum number of iterations during ICA fit.
    """

    # For the sake of example we annotate first 10 seconds of the recording as
    # 'BAD'. This part of data is excluded from the ICA decomposition by default.
    # To turn this behavior off, pass ``reject_by_annotation=False`` to
    # :meth:`mne.preprocessing.ICA.fit`.
    if bad_seconds is not None:
        raw.annotations = mne.Annotations([0], [bad_seconds], 'BAD')



    ica = ICA(n_components=n_components, method=method, random_state=random_state,
              verbose='warning', max_iter=max_iter)


    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                               stim=False, exclude='bads')

    #--------------------------------------------------------------------------
    # Fit ICA
    #--------------------------------------------------------------------------
    ica.fit(raw, picks=picks_meg, decim=decim, reject=reject)

    #--------------------------------------------------------------------------
    # Advanced artifact detection
    #--------------------------------------------------------------------------

    # EOG
    # eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
    # eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation
    eog_inds, scores = ica.find_bads_eog(raw)
    #
    eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
    n_eog_epochs = len(eog_epochs)

    if plot:
        eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
        eog_average = create_eog_epochs(raw, reject=dict(mag=5e-12, grad=4000e-13),
                                        picks=picks_meg).average()
        ica.plot_scores(scores, exclude=eog_inds, show=False)
        ica.plot_sources(eog_average, exclude=eog_inds, show=False)
        ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
        if len(eog_inds) > 0:
            ica.plot_properties(eog_epochs, picks=eog_inds[0], psd_args={'fmax': 35.},
                                image_args={'sigma': 1.}, show=False)



    # ECG
    ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks_meg)
    ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')  # find via correlation #, threshold = 0.125

    if plot:
        #
        ecg_average = create_ecg_epochs(raw, reject=dict(mag=5e-12, grad=4000e-13),
                                        picks=picks_meg).average()
        ica.plot_scores(scores, exclude=ecg_inds, show=False)
        ica.plot_sources(ecg_average, exclude=ecg_inds, show=False)
        ica.plot_overlay(ecg_average, exclude=ecg_inds, show=False)
        if len(ecg_inds) > 0:
            ica.plot_properties(ecg_epochs, picks=ecg_inds[0], psd_args={'fmax': 35.},
                                image_args={'sigma': 1.}, show=False)

    if plot:
        plt.show()

    # Exluce bad components
    ica.exclude.extend(eog_inds[:n_max_eog])
    ica.exclude.extend(ecg_inds[:n_max_ecg])

    # uncomment this for reading and writing
    # ica.save('my-ica.fif')
    # ica = read_ica('my-ica.fif')
    return ica


def preprocess_raw(raw, subject, kind, n_components = 0.99, fmin = 0.08, fmax = None, plot = False):
    """
    Preprocess raw data:
        - Apply Maxwell filter
        - Highpass filter at fmin
        - Fit and apply ICA

    Before applying ICA, a copy of raw is done.

    Args:
        raw
        subject
        kind
        fmin, fmax  : frequencies of bandpass filter before ICA
    """
    mne.channels.fix_mag_coil_types(raw.info)

    # Apply maxfilter
    raw = _run_maxfilter(raw, subject, kind)

    # 1Hz high pass is often helpful for fitting ICA
    raw.filter(fmin, fmax, n_jobs=1, fir_design='firwin')

    # Apply ICA
    ica = get_ica(raw, n_components = n_components,  plot = plot)

    if plot:
        plt.show()

    raw_copy = raw.copy()
    ica.apply(raw_copy)
    return raw_copy, ica


def get_raw(subject, kind):
    # Get raw
    fname = op.join(
            cfg.camcan_meg_raw_path,
            subject, kind, '%s_raw.fif' % kind)
    raw = mne.io.read_raw_fif(fname)
    return raw
#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == '__main__':

    # Select subject and condition
    subject_ind = 400 #485
    subject = subjects[subject_ind]
    kind    = 'task'

    # Get raw
    raw = get_raw(subject, kind)

    # prepocess
    raw_, ica = preprocess_raw(raw, subject, kind, plot = True)



    #################################
    plt.figure(1)
    data = raw.get_data()


    signal_1 = data[0, 100000:101000]
    plt.plot(signal_1)

    signal_2 = data[0, 0:1000]
    plt.plot(signal_2)


    #################################
    plt.figure(2)
    data = raw_.get_data()

    signal_1 = data[0, 100000:101000]
    plt.plot(signal_1)

    signal_2 = data[0, 0:1000]
    plt.plot(signal_2)

    plt.show()


    #===========================================================================
    #===========================================================================

    # raw_copy = raw.copy()
    # ica.apply(raw_copy)


    # #  Add ssp projection vectors
    # _compute_add_ssp_exg(raw)

    # Notes:
    # pick_types
