import os
import pickle

import apogee.select as apsel
import apogee.tools.read as apread

from . import utils

# each funct either loads file or calculates and saves it if it is not created already. onCluster affects only where files are looked for, I don't expect to use this

if __name__ == '__main__':
    # call module as script to ensure all pickles are created
    get_apo()
    get_allStar()
    get_allStar_statIndx()


def get_apo(onCluster=True):
    """
    """
    dataDir = utils.clusterDataDir if onCluster else utils.localDataDir
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf.dat')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            apo = pickle.load(f)
    else:
        apo = apsel.apogeeCombinedSelect()
        with open(path, 'wb') as f:
            pickle.dump(apo, f)
    # maybe add del line here or later if I have memory problems
    return apo

def get_allStar(onCluster=True):
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
    dataDir = utils.clusterDataDir if onCluster else utils.localDataDir
    path = os.path.join(dataDir, 'input_data', 'dr16allStar.dat')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            allStar = pickle.load(f)
    else:
        print('WARNING: calculating allStar')
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

def get_allStar_statIndx(onCluster=True):
    """
    returns both as whenever statIndx is needed, allStar is too
    """
    allStar = get_allStar()
    dataDir = utils.clusterDataDir if onCluster else utils.localDataDir
    statIndxPath = os.path.join(dataDir, 'input_data', 'dr16statIndx.dat')
    if os.path.exists(statIndxPath):
        with open(statIndxPath, 'rb') as f:
            statIndx = pickle.load(f)
    else:
        print('WARNING: calculating statIndx')
        apo = get_apo()
        statIndx = apo.determine_statistical(allStar)
        with open(path, 'wb') as f:
            pickle.dump(statIndx, f)
    assert np.count_nonzero(statIndx)==165768
    return (allStar, statIndx)
