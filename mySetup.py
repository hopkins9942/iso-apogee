from math import isclose
import numpy as np
import astropy.coordinates as coord
import astropy.units as u


dataDir = '/Users/hopkinsm/data/APOGEE/'


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
    
    
binsChoice = 0
# use this in other script to extract default bins
#Leaves option to access other sets of bins if wanted
# eg:
# FeHEdges, MgFeEdges, dicts = binsPossibilities[binsChoice]
binsPossibilities = [
    arr((-1.975, 0.725, 0.1))
    
    ]


#Reminder: kroupa isochrones are at MH = -1.975, -1.925, -1.875, ..., 0.625