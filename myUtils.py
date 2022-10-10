from math import isclose
import numpy as np
import astropy.coordinates as coord
import astropy.units as u


# paths - edit to match machines used
# Assumes makeBins.py, calcEffSelFunct.py and doFit.py are run on a cluster,
# data directory is scp-ed to local machine then analysis.py is run there
clusterDataDir = '/data/phys-galactic-isos/sjoh4701/APOGEE/' # used by makeBins.py, calcEffSelFunct.py and doFit.py
localDataDir = '/Users/hopkinsm/data/APOGEE/' # used by analysis.py and 

GC_frame = coord.Galactocentric() #adjust parameters here if needed
z_Sun = GC_frame.z_sun.to(u.kpc).value # .value removes unit, which causes problems with pytorch
R_Sun = np.sqrt(GC_frame.galcen_distance.to(u.kpc).value**2 - z_Sun**2)

muMin = 4.0
muMax = 17.0
muStep = 0.1
muGridParams = (muMin, muMax, muStep)

def arr(gridParams):
    start, stop, step = gridParams
    arr = np.arange(round((stop-start)/step)+1)*step+start
    assert isclose(arr[-1],stop) # will highlight both bugs and when stop-start is not multiple of diff
    return arr

def binName(binDict):
    binDict = dict(sorted(binDict.items()))
    # ensures same order independant of construction. I think sorts according to ASCII value of first character of key
    return '_'.join(['_'.join([key, f'{limits[0]:.3f}', f'{limits[1]:.3f}']) for key,limits in binDict.items()])
    #Note: python list comprehensions are cool

_FeH_edges = arr((-1.975, 0.725, 0.1)) #0.625-09.725 has no APOGEE statsample stars, -1.575--1.475 has about 130
# isochfrones have MH from -2.0 to 0.65 every 0.05, with one at 0.69525
#binsToUse = [{'FeH': (_FeH_edges[i], _FeH_edges[i+1])} for i in range(len(_FeH_edges)-1)]
# 27 bins

#age_FeH
_age_edges = np.array([0.0,4.5,9.0,14.0]) # remember ages only good for FeH>-0.5
_FeH_edges_for_age = arr((-0.475, 0.725, 0.1))
#binsToUse = [{'FeH': (_FeH_edges_for_age[i], _FeH_edges_for_age[i+1]), 'age': (_age_edges[j], _age_edges[j+1])} for i in range(len(_FeH_edges_for_age)-1) for j in range(len(_age_edges)-1)]
# 36 bins

# MgFe_FeH
_MgFe_edges = arr((0.0,0.5,0.05))

_FeH_edges_for_MgFe = arr((-1.375, 0.525, 0.1))
binsToUse = [{'FeH': (_FeH_edges_for_MgFe[i], _FeH_edges_for_MgFe[i+1]), 'MgFe': (_MgFe_edges[j], _MgFe_edges[j+1])} for i in range(len(_FeH_edges_for_MgFe)-1) for j in range(len(_MgFe_edges)-1)]
# 96 bins


