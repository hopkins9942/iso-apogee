import os
import pickle
import numpy as np

from mySetup import dataDir

def get_locations():
    path = os.path.join(dataDir, 'input_data', 'locations.dat')
    with open(path, 'rb') as f:
        locations = pickle.load(f)
    return locations


def get_solidAngles():
    path = os.path.join(dataDir, 'input_data', 'solidAngles.dat')
    with open(path, 'rb') as f:
        solidAngles = pickle.load(f)
    return solidAngles

def get_gLongLat():
    path = os.path.join(dataDir, 'input_data', 'gLongLat.dat')
    with open(path, 'rb') as f:
        gLongLat = pickle.load(f)
    return gLongLat


def get_allStar():
    """
    Options:
        rmcommissioning=True,
        main=True,
        exclude_star_bad=True,
        exclude_star_warn=True,
        use_astroNN_distances=True,
        use_astroNN_ages=True,
        rmdups=True
    """
    path = os.path.join(dataDir, 'input_data', 'dr16allStar.dat')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            allStar = pickle.load(f)
    assert len(allStar) == 211051
    return allStar



def get_statIndx():
    """
    """
    statIndxPath = os.path.join(dataDir, 'input_data', 'dr16statIndx.dat')
    with open(statIndxPath, 'rb') as f:
        statIndx = pickle.load(f)
    assert np.count_nonzero(statIndx)==165768
    return statIndx



def get_statSample():
    """
    With columns
    ['LOCATION_ID', 'FIELD', 
     'J', 'J_ERR', 'H', 'H_ERR', 'K', 'K_ERR',
     'GLON', 'GLAT',
     'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR',
     'M_H', 'M_H_ERR', 'ALPHA_M', 'ALPHA_M_ERR',
     'O_FE', 'MG_FE', 'SI_FE', 'S_FE', 'CA_FE', 'FE_H', 
     'O_FE_ERR', 'MG_FE_ERR', 'SI_FE_ERR', 'S_FE_ERR', 'CA_FE_ERR', 'FE_H_ERR',
     'weighted_dist', 'weighted_dist_error',
     'age_lowess_correct',
     'J0', 'H0', 'K0', 'METALS', 'ALPHAFE']
    """
    path = os.path.join(dataDir, 'input_data', 'dr16statSample.dat')
    with open(path, 'rb') as f:
        statSample = pickle.load(f)
    assert len(statSample)==165768
    assert len(statSample.dtype)==38
    return statSample

if __name__=='__main__':
    get_locations()
    get_solidAngles()
    get_gLongLat()
    get_allStar()
    get_statIndx()
    get_statSample()
