import multiprocessing
import pickle
import os
import sys
from functools import partial

import numpy as np
import apogee.select as apsel
import mwdust

import mySetup
import apogeePickles
import pickleGetters
import myIsochrones



def main():
    Ncpus = int(sys.argv[2])
    jobIndex = int(sys.argv[1])
    
    
    isogrid = myIsochrones.loadGrid()
    weights = myIsochrones.calcWeights(isogrid)
    
    MH_logAge, indices = myIsochrones.extractIsochrones(isogrid)
    MH, logAge = MH_logAge[jobIndex]
    
    if jobIndex<len(indices):
        isoIndices = np.arange(indices[jobIndex], indices[jobIndex+1]) 
    else:
        isoIndices = np.arange(indices[jobIndex], len(isogrid))
    isochroneMask = np.zeros(len(isogrid), dtype=bool)
    isochroneMask[isoIndices] = True
    
        
        
    RGmask = myIsochrones.makeRGmask(isogrid)
    
    
    apo = apogeePickles.get_apo_lite()
    
    mu = mySetup.arr(mySetup.muGridParams)
    print(mu)
    D = 10**(-2 + 0.2*mu)
    
    locations = pickleGetters.get_locations()
    
    dmap = mwdust.Combined19(filter='2MASS H')
    apof = apsel.apogeeEffectiveSelect(apo, dmap3d=dmap,
                                       MH =isogrid[isochroneMask & RGmask]['Hmag'],
                                       JK0=isogrid[isochroneMask & RGmask]['Jmag']-isogrid[isochroneMask & RGmask]['Ksmag'],
                                       weights=weights[isochroneMask & RGmask])
    
    effSelFunc_mapper = partial(effSelFunc_helper, apof, D, locations)
    print("about to start multiprocessing")
    with multiprocessing.Pool(Ncpus) as p:
        print(f"parent: {os.getppid()}, child: {os.getpid()}")
        temp_effSelFunc = list(p.map(effSelFunc_mapper, range(len(locations)), chunksize=int(len(locations)/(4*Ncpus))))
    effSelFunc = np.array(temp_effSelFunc)
    
    print("Finished multiprocessing")
    
    filePath = os.path.join(mySetup.dataDir, 'ESF', f'MH_{MH:.3f}_logAge_{logAge:.3f}.dat')
    print(filePath)
    with open(filePath, 'wb') as f:
        pickle.dump(effSelFunc, f)
    
    
def effSelFunc_helper(apof, D, locations, i):
    """
    Needed as multiprocessed functions need to be defined at top level
    """
    return apof(locations[i], D)

if __name__=='__main__':
    main()
