"""
This file contains information about multifractal analysis parameters,
output folder for mfanalysis results and other useful parameters.
"""


import sys, os
from os.path import dirname, abspath
import numpy as np

def get_io_info():
    """
    Return information about input/output directories
    """
    # current folder
    current_dir = dirname(abspath(__file__))
    # project folder
    project_dir = dirname(dirname(current_dir))
    # output folder
    camcan_output_dir = os.path.join(project_dir, 'data_mf_out', 'camcan')

    mf_io_info = {}
    mf_io_info['camcan_output_dir'] = camcan_output_dir

    return mf_io_info


def get_mf_params():
    """
    Returns a list of dictionaries. Each diciionary contains the fields,
    corresponding to parameters of MF analysis:
        - 'wt_name'
        - 'formalism' # multifractal formalism
        - 'p'         # value of p for p-Leaders
        - 'j1'
        - 'j2'
        - 'n_cumul'
        - 'gamint'
        - 'wtype'


    """
    mf_params = []

    # new param - wlmf
    param = {}
    param['wt_name']   = 'db3'
    param['formalism'] = None
    param['p']         = np.inf
    param['j1']        = 9
    param['j2']        = 13
    param['n_cumul']   = 3
    param['gamint']    = 1.0
    param['wtype']     = 0
    mf_params.append(param)

    # new param - p Leaders, p = 1
    param = {}
    param['wt_name']   = 'db3'
    param['formalism'] = None
    param['p']         = 1.0
    param['j1']        = 9
    param['j2']        = 13
    param['n_cumul']   = 3
    param['gamint']    = 1.0
    param['wtype']     = 0
    mf_params.append(param)

    # new param, p Leaders, p = 2
    param = {}
    param['wt_name']   = 'db3'
    param['formalism'] = None
    param['p']         = 2.0
    param['j1']        = 9
    param['j2']        = 13
    param['n_cumul']   = 3
    param['gamint']    = 1.0
    param['wtype']     = 0
    mf_params.append(param)


    # new param, wcmf
    param = {}
    param['wt_name']   = 'db3'
    param['formalism'] = 'wcmf'
    param['p']         = None
    param['j1']        = 9
    param['j2']        = 13
    param['n_cumul']   = 3
    param['gamint']    = 1.0
    param['wtype']     = 0
    mf_params.append(param)


    return mf_params
