import sys
import os
import pickle
from functools import partial
import multiprocessing
import schwimmbad

import numpy as np

import apogee
import apogee.select as apsel
import mwdust
import tqdm
import isochrones as iso

_ROOTDIR = "/home/sjoh4701/APOGEE/iso-apogee/"



if os.path.exists(_ROOTDIR+'sav/apodr16_csf.dat'):
    with open(_ROOTDIR+'sav/apodr16_csf.dat', 'rb') as f:
        apo = pickle.load(f)
else:
    apo = apsel.apogeeCombinedSelect()
    with open(_ROOTDIR+'sav/apodr16_csf.dat', 'wb') as f:
        pickle.dump(apo, f)


FeHBinEdges = [float(sys.argv[1]), float(sys.argv[2])]
print(FeHBinEdges)
muMin = 0.0
muMax = 15 # CHECK THIS against allStar - better to exclude fringe datapoints than have all data in nearest three bins - plot mu distribution
muDiff = 0.1
muGridParams = (muMin, muMax, int((muMax-muMin)//muDiff)) # (start,stop,size)
mu = np.linspace(*muGridParams)
D = 10**(-2+0.2*mu) #kpc

locations = apo.list_fields(cohort='all')

isogrid = iso.newgrid()
# newgrid ensures means it uses new isochrones. I should either rewrite isochrones.py, maybe with MIST isochrones, or at least fully understand it
mask = ((isogrid['logg'] > 1) & (isogrid['logg'] < 3)
        & (isogrid['MH'] >  FeHBinEdges[0])
        & (isogrid['MH'] <= FeHBinEdges[1]))

dmap = mwdust.Combined19(filter='2MASS H')
# see Ted's code for how to include full grid
apof = apsel.apogeeEffectiveSelect(apo, dmap3d=dmap,
                                   MH=isogrid[mask]['Hmag'],
                                   JK0=isogrid[mask]['Jmag']-isogrid[mask]['Ksmag'],
                                   weights=isogrid[mask]['weights'])

def effSelFunct_helper(apof, D, locations, i):
    """
    Needed as multiprocessed functions need to be defined at top level
    """
    print(i)
    return apof(locations[i], D)


effSelFunct_mapper = partial(effSelFunct_helper, apof, D, locations)
with multiprocessing.Pool(6) as p:
    print("starting multiprocessing")
    temp_effSelFunct = list(p.map(effSelFunct_mapper, range(len(locations))))
effSelFunct = np.array(temp_effSelFunct)

            # this arcane series of tensors, arrays, lists and maps is because 
            # a list is because tensors are best constructed out of a single
            # array rather than a list of arrays, and neither np.array nor
            # torch.tensor know how to deal directly with a map object

filePath = (_ROOTDIR + "sav/EffSelGrids/" +
                    '_'.join([str(FeHBinEdges[0]), str(FeHBinEdges[1]), str(mu[0]),
                              str(mu[-1]), str(len(mu))])
                    + ".dat")

with open(filePath, 'wb') as f:
    pickle.dump(effSelFunct, f)



