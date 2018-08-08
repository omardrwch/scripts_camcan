import os.path as op

camcan_path = '/storage/store/data/camcan'
camcan_meg_path = op.join(
    camcan_path, 'camcan47/cc700/meg/pipeline/release004/')
camcan_meg_raw_path = op.join(camcan_meg_path, 'data/aamod_meg_get_fif_00001')

mne_camcan_freesurfer_path = (
    '/storage/store/data/camcan-mne/freesurfer')


# derivative_path = '/storage/store/derivatives/camcan/pipelines/base2018/MEG'
project_dir = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
output_path = op.join(project_dir, 'data_camcan_out')
