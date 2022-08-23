import os
from functools import partial
import pickle

import numpy as np
import matplotlib.pyplot as plt

import myUtils

# for 1D bins - assumed ordered, non-overlapping, equal size


def main():
    binList = myUtils.binsToUse
    
    FeHEdges = np.append([binList[i]['FeH'][0] for i in range(len(binList))],
                         binList[-1]['FeH'][1])
    FeHMidpoints = (FeHEdges[:-1] + FeHEdges[1:])/2
    FeHWidths = FeHEdges[1:] - FeHEdges[:-1]
    
    FeHnumberlogA  = np.zeros(len(binList))
    aR       = np.zeros(len(binList))
    az       = np.zeros(len(binList))
    NRG2mass = np.zeros(len(binList))
    
    for i, binDict in enumerate(binList):
        FeHnumberlogA[i], aR[i], az[i] = loadFitResults(binDict)
        NRG2mass[i] = loadNRG2mass(binDict)
    
    FeHmasslogA = FeHnumberlogA + np.log(NRG2mass)
    
    _FeH_p = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4])
    _fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
    comp = partial(np.interp, xp=_FeH_p, fp=_fH2O_p)
    
    fH2OEdges = myUtils.arr((0.06, 0.51, 0.01))
    fH2OMidpoints = (fH2OEdges[:-1] + fH2OEdges[1:])/2
    fH2OWidths = fH2OEdges[1:] - fH2OEdges[:-1]
    fH2OlogA = getfH2OlogA(fH2OEdges, FeHEdges, FeHmasslogA)
    
    #plotting
    fig, ax = plt.subplots()
    ax.bar(FeHMidpoints, FeHmasslogA, width = FeHWidths)
    
    fig, ax = plt.subplots()
    ax.bar(fH2OMidpoints, fH2OlogA, width= fH2OWidths)
    for i, FeH in enumerate(FeHEdges):
        ax.plot([comp(FeH),comp(FeH)], [0,1e6])
    
    
    

binsDir = '/Users/hopkinsm/data/APOGEE/bins'
    
_FeH_p = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4])
_fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
comp = partial(np.interp, xp=_FeH_p, fp=_fH2O_p)

#def binIndex(FeH):
#    #For 1D bins only
#    return np.nonzero(edges<=FeH)[0][-1]

def loadFitResults(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'fit_results.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

def loadNRG2mass(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'NRG2mass.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

def getfH2OlogA(fH2OEdges, FeHEdges, FeHmasslogA):
    """assumes FeH bins equal size"""
    alpha = 1 # unknown constant of proportionality between number of ISOs produces and stellar mass
    #FeH_step = FeHEdges[1] - FeHEdges[0]
    fH2OlogA = np.zeros(len(fH2OEdges)-1)
    NpointsPerBin = int(20*(FeHEdges[1]-FeHEdges[0])/(fH2OEdges[1]-fH2OEdges[0]))
    
    FeHarray = np.linspace(FeHEdges[0], FeHEdges[-1], NpointsPerBin*(len(FeHEdges)-1) + 1)
    FeHarray = (FeHarray[:-1] + FeHarray[1:])/2
    # evenly spaces NpointsPerBin points in each bin offset from edges
    fH2Oarray = comp(FeHarray)
    
    for j in range(len(fH2OlogA)):
        indexArray = np.nonzero((fH2OEdges[j] <= fH2Oarray)&(fH2Oarray < fH2OEdges[j+1]))[0]//NpointsPerBin
        # finds index of points in FeHarray, integer division changes them to index in FeHEdges bins
        fH2OlogA[j] += ((alpha/NpointsPerBin) * np.exp(FeHmasslogA[indexArray])).sum()
    return fH2OlogA
    
    
        
    
    
    
# class oneDdensityModel:
#     def __init__(self, binEdges, logAmp, aR, az):
#         self.binEdges = binEdges
#         self.logAmp = logAmp
#         self.aR = aR
#         self.az = az
            
#     def binIndex(value):
#     """
#     takes kwargs of values, finds bin those values lie in
#     Assumes no overlapping bins
#     """
#     for i, bin in enumerate(self.binList):
#         isIn = True
#         for key, limits in bin.items():
#             isIn &= (limits[0]<=kwargs[key]) & (kwargs[key]<limits[1])
#         if isIn: return i
#     return np.nan # not in any bin

#     def distribution(paramSpacePoint, position, normalised=False):
#         R, z = position
#         i = self.binIndex(**paramSpacePoint)
#         return np.exp(self.logAmp[i] - self.aR[i]*(R-myUtils.R_Sun) - self.az[i]*np.abs(z))
    
#     def integrated(paramSpacePoints, normalised=False):)
#         i = self.binIndex(**paramSpacePoint)
#         return 4*np.pi*np.exp(self.logAmp[i] + self.aR[i]*myUtils.R_Sun)/(self.aR[i]**2 * self.az[i])

# def binParamSpaceVol(binDict):
#     size=1
#     for limits in binDict.values():
#         size*=(limits[1]-limits[0])
#     return size
    
    
if __name__=='__main__':
    main()
