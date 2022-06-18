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

import DensityModelling_defs as dm

_ROOTDIR = "/home/sjoh4701/APOGEE/iso-apogee/"
_NCPUS = int(sys.argv[2])

JOB_INDEX = int(sys.argv[1])

if os.path.exists(_ROOTDIR+'sav/apodr16_csf.dat'):
    with open(_ROOTDIR+'sav/apodr16_csf.dat', 'rb') as f:
        apo = pickle.load(f)
else:
    apo = apsel.apogeeCombinedSelect()
    with open(_ROOTDIR+'sav/apodr16_csf.dat', 'wb') as f:
        pickle.dump(apo, f)
del apo._specdata, apo._photdata, apo.apo1sel._specdata, apo.apo1sel._photdata, apo.apo2Nsel._specdata, apo.apo2Nsel._photdata, apo.apo2Ssel._specdata, apo.apo2Ssel._photdata

FeHBinParams = (-1.025, 0.475, 0.1)
edgesArray = dm.arr(FeHBinParams)
print(edgesArray)

FeHBinEdges = [edgesArray[JOB_INDEX], edgesArray[JOB_INDEX+1]]
print(FeHBinEdges)
#muMin = 4.0
#muMax = 17.0 # allStar statistical sample mu distribution is approximately between 3.3-17.3
#muStep = 0.1
#muGridParams = (muMin, muMax, muStep) # (start,stop,step)
mu = dm.arr(dm.muGridParams)
print(mu)
D = 10**(-2+0.2*mu) #kpc

locations = apo.list_fields(cohort='all')
print(len(locations))
isogrid = iso.newgrid()
# newgrid ensures means it uses new isochrones. I should either rewrite isochrones.py, maybe with MIST isochrones, or at least fully understand it
mask = ((1 <= isogrid['logg']) & (isogrid['logg'] < 3)
        & (FeHBinEdges[0] <= isogrid['MH'])
        & (isogrid['MH'] < FeHBinEdges[1]))

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
    return apof(locations[i], D)


effSelFunct_mapper = partial(effSelFunct_helper, apof, D, locations)
print("about to start multiprocessing")
with multiprocessing.Pool(_NCPUS) as p:
    print(f"parent: {os.getppid()}, child: {os.getpid()}")
    temp_effSelFunct = list(tqdm.tqdm(p.map(effSelFunct_mapper, range(len(locations)), chunksize=int(len(locations)/(4*_NCPUS))), total=len(locations)))
effSelFunct = np.array(temp_effSelFunct)

print("Finished multiprocessing")

            # this arcane series of tensors, arrays, lists and maps is because 
            # a list is because tensors are best constructed out of a single
            # array rather than a list of arrays, and neither np.array nor
            # torch.tensor know how to deal directly with a map object

filePath = (_ROOTDIR + "sav/EffSelFunctGrids/" +
                    '_'.join(["FeH", f'{FeHBinEdges[0]:.3f}', f'{FeHBinEdges[1]:.3f}'])
                    + ".dat")

with open(filePath, 'wb') as f:
    pickle.dump(effSelFunct, f)



