import multiprocessing
import pickle
import os
import sys
from functools import partial

import numpy as np
import apogee.select as apsel
import mwdust

from myUtils import binsToUse, binName, clusterDataDir, arr, muGridParams, fitLim
import pickleGetters
import isochrones
import makeBins



# TODO: use fact that MgFe doesn't affect effselfuct to speed up - ie copy just fe bin one?
def main():
    Ncpus = int(sys.argv[2])
    jobIndex = int(sys.argv[1])
    
    binDict = binsToUse[jobIndex]
    print(binDict)
    
    binPath = os.path.join(clusterDataDir, 'bins', binName(binDict))
    
    with open(os.path.join(binPath, 'data.dat'), 'rb') as f:
        data = pickle.load(f)
#    if data[0]<myUtils.fitLim: # No stars in bin, no need to fit
# needs fixing - low star bins are good to look at, and data[0] isn't an int due to bad FeH distribution
        assert False
        # put in somethiong here so doesn't recalculate same effselfunc for every Mg bin at same Fe

        effSelFunc = np.zeros(1)
        print("No stars, not fit")
        filePath = os.path.join(binPath, 'effSelFunc.dat')
        print(filePath)
        with open(filePath, 'wb') as f:
            pickle.dump(effSelFunc, f)
        return 0
    
    
    
    apo = pickleGetters.get_apo()
    del apo._specdata, apo._photdata, apo.apo1sel._specdata, apo.apo1sel._photdata, apo.apo2Nsel._specdata, apo.apo2Nsel._photdata, apo.apo2Ssel._specdata, apo.apo2Ssel._photdata
    # apo deep copied in each process, this saves memory
    
    
    mu = arr(muGridParams)
    print(mu)
    D = 10**(-2 + 0.2*mu)
    
    locations = apo.list_fields(cohort='all')
    print(len(locations))
    isogrid = isochrones.newgrid()
    # newgrid ensures means it uses new isochrones. I should either rewrite isochrones.py, maybe with MIST isochrones, or at least fully understand it
    # and with isochrone utilities, to avoid following weird import
     
    mask = makeBins.calc_isogrid_mask(binDict,isogrid)
    dmap = mwdust.Combined19(filter='2MASS H')
    apof = apsel.apogeeEffectiveSelect(apo, dmap3d=dmap,
                                       MH=isogrid[mask]['Hmag'],
                                       JK0=isogrid[mask]['Jmag']-isogrid[mask]['Ksmag'],
                                       weights=isogrid[mask]['weights'])
    
    effSelFunc_mapper = partial(effSelFunc_helper, apof, D, locations)
    print("about to start multiprocessing")
    with multiprocessing.Pool(Ncpus) as p:
        print(f"parent: {os.getppid()}, child: {os.getpid()}")
        temp_effSelFunc = list(p.map(effSelFunc_mapper, range(len(locations)), chunksize=int(len(locations)/(4*Ncpus))))
    effSelFunc = np.array(temp_effSelFunc)
    
    print("Finished multiprocessing")
    
    filePath = os.path.join(binPath, 'effSelFunc.dat')
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
