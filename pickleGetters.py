import os
import pickle

import numpy as np

from myUtils import dataDir
#sticking with .pth file method as allows files to be reached interactively and relative imports only work in subpackages

# each funct either loads file or calculates and saves it if it is not created already. onCluster affects only where files are looked for, I don't expect to use this



def get_apo():
    """
    """
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf.dat')
    with open(path, 'rb') as f:
        apo = pickle.load(f)
    return apo

def get_apo_lite():
    """
    """
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf_lite.dat')
    with open(path, 'rb') as f:
        apo_lite = pickle.load(f)
    return apo_lite



def get_allStar():
    """
    uses a pickle with options:
        rmcommissioning=True,
        main=True,
        exclude_star_bad=True,
        exclude_star_warn=True,
        use_astroNN_distances=True,
        use_astroNN_ages=True,
        rmdups=True
    """
    path = os.path.join(dataDir, 'input_data', 'dr16allStar.dat')
    with open(path, 'rb') as f:
        allStar = pickle.load(f)
    assert len(allStar) == 211051
    return allStar

def get_allStar_statIndx():
    """
    returns both as whenever statIndx is needed, allStar is too
    """
    allStar = get_allStar()
    statIndxPath = os.path.join(dataDir, 'input_data', 'dr16statIndx.dat')
    with open(statIndxPath, 'rb') as f:
        statIndx = pickle.load(f)
    assert np.count_nonzero(statIndx)==165768
    return (allStar, statIndx)

    
    