import os
from math import isclose

import numpy as np
import astropy.coordinates as coord
import astropy.units as u


#dataDir = '/Users/hopkinsm/data/APOGEE/'
dataDir = '/data/phys-galactic-isos/sjoh4701/APOGEE/'


GC_frame = coord.Galactocentric() #adjust parameters here if needed
z_Sun = GC_frame.z_sun.to(u.kpc).value # .value removes unit, which causes problems with pytorch
R_Sun = np.sqrt(GC_frame.galcen_distance.to(u.kpc).value**2 - z_Sun**2)


muMin = 7.0 # only one RG below 7.0
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
    
def D2mu(D):
    return 10+5*np.log10(D)
    
def mu2D(mu):
    return 10**(-2+0.2*mu)


# actual
FeHEdges = arr((-2.0, 0.7, 0.1))
aFeEdges = arr((-0.2, 0.5, 0.1))



binList = [
    {
    'FeH': (FeHEdges[i], FeHEdges[i+1]),
    'aFe': (aFeEdges[j], aFeEdges[j+1])
    } for i in range(len(FeHEdges)-1) for j in range(len(aFeEdges)-1)]
#this is not general, but I'm only doing one thing


#Reminder: kroupa isochrones are at MH = -1.975, -1.925, -1.875, ..., 0.675
# and logAge = 

if __name__=='__main__':
    for binDict in binList:
        # creates bin directory
        name = binName(binDict)
        print(name)
        path = os.path.join(dataDir, 'bins', name)
        if not os.path.exists(path):
            os.makedirs(path)
