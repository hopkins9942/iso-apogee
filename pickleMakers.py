import os
import pickle

import numpy as np
import apogee.select as apsel
import apogee.tools.read as apread

from setup import dataDir


# Idea: call, if already made will just load unless force=True


def make_apo(force=False):
    """
    """
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf.dat')
    
    if os.path.exists(path) and (not force):
        print("apo already calculated!")
        with open(path, 'rb') as f:
            apo = pickle.load(f)
            
    else:
        apo = apsel.apogeeCombinedSelect()
        with open(path, 'wb') as f:
            pickle.dump(apo, f)
            
    return apo


def make_apo_lite(force=False):
    """
    """
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf_lite.dat')
    
    if os.path.exists(path) and (not force):
        print("apo_lite already calculated!")
        with open(path, 'rb') as f:
            apo = pickle.load(f)
            
    else:
        apo = make_apo(force)
        del apo._specdata, apo._photdata, apo.apo1sel._specdata, apo.apo1sel._photdata, apo.apo2Nsel._specdata, apo.apo2Nsel._photdata, apo.apo2Ssel._specdata, apo.apo2Ssel._photdata
        with open(path, 'wb') as f:
            pickle.dump(apo, f)
    
    return apo


def make_allStar(force=False):
    """
    makes a pickle with options:
        rmcommissioning=True,
        main=True,
        exclude_star_bad=True,
        exclude_star_warn=True,
        use_astroNN_distances=True,
        use_astroNN_ages=True,
        rmdups=True
    """
    path = os.path.join(dataDir, 'input_data', 'dr16allStar.dat')
    
    if os.path.exists(path) and (not force):
        with open(path, 'rb') as f:
            print("allStar already calculated!")
            allStar = pickle.load(f)
            
    else:
        allStar = apread.allStar(
            rmcommissioning=True,
            main=True,
            exclude_star_bad=True,
            exclude_star_warn=True,
            use_astroNN_distances=True,
            use_astroNN_ages=True,
            rmdups=True)
        with open(path, 'wb') as f:
            pickle.dump(allStar, f)
            
    assert len(allStar) == 211051
    return allStar


def make_allStar_statIndx(force=False):
    """
    returns both as whenever statIndx is needed, allStar is too
    """
    allStar = make_allStar(force)
    statIndxPath = os.path.join(dataDir, 'input_data', 'dr16statIndx.dat')
    
    if os.path.exists(statIndxPath) and (not force):
        with open(statIndxPath, 'rb') as f:
            print("statIndex already calculated!")
            statIndx = pickle.load(f)
    
    else:
        apo = make_apo(force)
        statIndx = apo.determine_statistical(allStar)
        with open(statIndxPath, 'wb') as f:
            pickle.dump(statIndx, f)
            
    assert np.count_nonzero(statIndx)==165768
    return (allStar, statIndx)



if __name__ == '__main__':
    # call module as script to ensure all pickles are created
    # Only calling some as some involve multiple pickles
    make_apo_lite(force=True)
    make_allStar_statIndx(force=True)
    
