import os
import pickle

import numpy as np
from numpy.lib.recfunctions import drop_fields 
import astropy.coordinates as coord
import astropy.units as u
import apogee.select as apsel
import apogee.tools.read as apread

from setup import dataDir


# Concept: call, if already made will just load unless force=True
# Concept: make_data makes pickles of .np objects for use anywhere,
#          not just where apogee importable


# consider moving and importing
# GC_frame = coord.Galactocentric() #adjust parameters here if needed
# z_Sun = GC_frame.z_sun.to(u.kpc).value # .value removes unit, which causes problems with pytorch
# R_Sun = np.sqrt(GC_frame.galcen_distance.to(u.kpc).value**2 - z_Sun**2)


# apogee pickles

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

def make_locations(force=False):
    
    WORK OUT WHAT TO DO WITH THESE AND DATA
    
def make_solidAngles(force=False):

    

def make_data(force=False):
    """
    makes pickles of field IDs and solid angles, allStar dists, angular coords
    abundances and ages as lists or np.arrays so they can be loaded where apogee is not
    importable.
    """
    
    path = os.path.join(dataDir, 'input_data', 'dr16statSample.dat')
    
    if os.path.exists(path) and (not force):
        with open(path, 'rb') as f:
            print("data already calculated!")
            S = pickle.load(f)
            
    else:
        apo = make_apo_lite(force)
        locations = apo.list_fields(cohort='all') # list
        solidAngles = [apo.area(loc)*(np.pi/180)**2 for loc in locations]
        
        allStar, statIndx = make_allStar_statIndx(force)
        statSample = allStar[statIndx] # statistical sample, np structured array
        
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
    
        S = drop_fields(allStar, names2drop)
        
        path = os.path.join(dataDir, 'input_data', 'dr16statSample.dat')
        with open(path, 'wb') as f:
            pickle.dump(S, f)
            
    assert len(S) == 165768
    return S
    
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
    
    
    
    

if __name__ == '__main__':
    # call module as script to ensure all pickles are created
    # Only calling some as some involve multiple pickles
    make_data(force=True)




