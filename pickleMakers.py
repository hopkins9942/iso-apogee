import os
import pickle

import numpy as np
import apogee.select as apsel
import apogee.tools.read as apread

from myUtils import dataDir
#sticking with .pth file method as allows files to be reached interactively and relative imports only work in subpackages

# each funct either loads file or calculates and saves it if it is not created already. 



def make_apo():
    """
    """
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf.dat')
    if os.path.exists(path):
        print("apo already calculated!")
        with open(path, 'rb') as f:
            apo = pickle.load(f)
    else:
        apo = apsel.apogeeCombinedSelect()
        with open(path, 'wb') as f:
            pickle.dump(apo, f)
    return apo

def make_apo_lite():
    """
    """
    lite_path = os.path.join(dataDir, 'input_data', 'apodr16_csf_lite.dat')
    apo_path = os.path.join(dataDir, 'input_data', 'apodr16_csf.dat')
    if os.path.exists(lite_path):
        print("apo_lite already calculated!")
        with open(lite_path, 'rb') as f:
            apo = pickle.load(f)
            
    elif os.path.exists(apo_path):
        print("using apo")
        with open(apo_path, 'rb') as f:
            apo = pickle.load(f)
        del apo._specdata, apo._photdata, apo.apo1sel._specdata, apo.apo1sel._photdata, apo.apo2Nsel._specdata, apo.apo2Nsel._photdata, apo.apo2Ssel._specdata, apo.apo2Ssel._photdata
        with open(lite_path, 'wb') as f:
            pickle.dump(apo, f)
            
    else:
        apo = apsel.apogeeCombinedSelect()
        del apo._specdata, apo._photdata, apo.apo1sel._specdata, apo.apo1sel._photdata, apo.apo2Nsel._specdata, apo.apo2Nsel._photdata, apo.apo2Ssel._specdata, apo.apo2Ssel._photdata
        with open(lite_path, 'wb') as f:
            pickle.dump(apo, f)
    
    return apo



def make_allStar():
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
    if os.path.exists(path):
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


def make_allStar_statIndx():
    """
    returns both as whenever statIndx is needed, allStar is too
    """
    allStar = make_allStar()
    statIndxPath = os.path.join(dataDir, 'input_data', 'dr16statIndx.dat')
    if os.path.exists(statIndxPath):
        with open(statIndxPath, 'rb') as f:
            print("statIndex already calculated!")
            statIndx = pickle.load(f)
    else:
        apo = make_apo()
        statIndx = apo.determine_statistical(allStar)
        with open(statIndxPath, 'wb') as f:
            pickle.dump(statIndx, f)
    assert np.count_nonzero(statIndx)==165768
    return (allStar, statIndx)



if __name__ == '__main__':
    # call module as script to ensure all pickles are created
    make_apo()
    make_apo_lite()
    make_allStar()
    make_allStar_statIndx()
    
    