import os
import pickle

import numpy as np
from numpy.lib.recfunctions import drop_fields 
import astropy.coordinates as coord
import astropy.units as u
import apogee.select as apsel
import apogee.tools.read as apread

from putStuffHere import dataDir
from pickleGetters import get_allStar, get_statIndx

# This file is for pickling the objects and data which require apogee.
# For apo and apo_lite, this file also contains the unpickling functions.
# This is so that the data which doesn't require apogee can be unpickled as
# np arrays in pickleGetters in a different location without apogee being installed.


# Concept: don't overcomplicate. Do what it says on tin. If object is needed by make, it uses get

# import and call make_all() to remake all pickles


# apogee pickles
def get_apo():
    """
    """
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf.dat')
    
    if os.path.exists(path):
        with open(path, 'rb') as f:
            apo = pickle.load(f)
    else:
        raise RuntimeError("apo not made, call make_apo()")
    return apo


def make_apo():
    """
    """
    print("Making apo")
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf.dat')
    apo = apsel.apogeeCombinedSelect()
    with open(path, 'wb') as f:
        pickle.dump(apo, f)
    return apo


def get_apo_lite():
    """
    """
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf_lite.dat')
    with open(path, 'rb') as f:
        apo_lite = pickle.load(f)
    return apo_lite


def make_apo_lite(remake = True, recursive=False):
    """
    """
    path = os.path.join(dataDir, 'input_data', 'apodr16_csf_lite.dat')
    apo = get_apo()
    del apo._specdata, apo._photdata, apo.apo1sel._specdata, apo.apo1sel._photdata, apo.apo2Nsel._specdata, apo.apo2Nsel._photdata, apo.apo2Ssel._specdata, apo.apo2Ssel._photdata
    with open(path, 'wb') as f:
        pickle.dump(apo, f)
    return apo


# none apogee objects


def make_locations_solidAngles():
    """
    """
    apo = get_apo_lite()
    locations = apo.list_fields(cohort='all') # list
    solidAngles = [apo.area(loc)*(np.pi/180)**2 for loc in locations]
    locpath = os.path.join(dataDir, 'input_data', 'locations.dat')
    sApath = os.path.join(dataDir, 'input_data', 'solidAngles.dat')
    with open(locpath, 'wb') as f:
        pickle.dump(locations, f)
    with open(sApath, 'wb') as f:
        pickle.dump(solidAngles, f)
    return (locations, solidAngles)

def make_allStar():
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


def make_statIndx():
    """
    """
    allStar = get_allStar()
    statIndxPath = os.path.join(dataDir, 'input_data', 'dr16statIndx.dat')
    apo = get_apo()
    statIndx = apo.determine_statistical(allStar)
    with open(statIndxPath, 'wb') as f:
        pickle.dump(statIndx, f)
    assert np.count_nonzero(statIndx)==165768
    return (allStar, statIndx)

def make_statSample(force=False):
    """
    makes pickles of field IDs and solid angles, allStar dists, angular coords
    abundances and ages as lists or np.arrays so they can be loaded where apogee is not
    importable.
    """
    
    path = os.path.join(dataDir, 'input_data', 'dr16statSample.dat')
    
    
    
    allStar = get_allStar()
    statIndx = get_statIndx()
    S = allStar[statIndx] # statistical sample, np structured array
    
    # pickling only relevant fields 
    names = allStar.dtype.names
    names2keep = ['LOCATION_ID', 'FIELD', 
                  'J', 'J_ERR', 'H', 'H_ERR', 'K', 'K_ERR',
                  'GLON', 'GLAT',
                  'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR',
                  'M_H', 'M_H_ERR', 'ALPHA_M', 'ALPHA_M_ERR'
                  'O_FE', 'MG_FE', 'SI_FE', 'S_FE', 'CA_FE', 'FE_H', 
                  'O_FE_ERR', 'MG_FE_ERR', 'SI_FE_ERR', 'S_FE_ERR', 'CA_FE_ERR', 'FE_H_ERR',
                  'weighted_dist', 'weighted_dist_error',
                  'age_lowess_correct',
                  'J0', 'H0', 'K0', 'METALS', 'ALPHAFE']
    
    names2drop = []
    for n in names:
        if not (n in names2keep):
            names2drop.append(n)

    statSample = drop_fields(S, names2drop)
    
    with open(path, 'wb') as f:
        pickle.dump(statSample, f)
        
    assert len(statSample) == 165768
    assert len(statSample.dtype) == 38
    return statSample
    
    # Could save individuals, but trimmed S is only order 10MB maybe
    # gLon = S['GLON'] # check what type these are - np.recarray? 
    # gLat = S['GLAT']
    # D = S['weighted_dist']/1000 # kpc
    # # gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
    # # gCentricCoords = gCoords.transform_to(GC_frame)
    # # x = gCentricCoords.x.to(u.kpc).value
    # # y = gCentricCoords.y.to(u.kpc).value
    # # z = gCentricCoords.z.to(u.kpc).value
    # # R = np.sqrt(x**2 + y**2)
    # # modz = np.abs(z)
    
    # FeH = S['FE_H']
    # OH = S['O_H']
    # MgH = S['MG_H']
    # SiH = S['SI_H']
    # SH = S['S_H']
    # CaH = S['CA_H']
    
def make_all():
    """
    Makes all pickles
    """
    print("making apo")
    make_apo()
    print("making apo_lite")
    make_apo_lite()
    print("making locations_solidAngles")
    make_locations_solidAngles()
    print("making allStar")
    make_allStar()
    print("making statIndx")
    make_statIndx()
    print("making statSample")
    make_statSample()
    print("done!")
    
    

if __name__ == '__main__':
    make_all()




