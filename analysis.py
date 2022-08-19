import os

imort numpy as np
import matplotlib.pyplot as plt

import myUtils

def main():
    binList = myUtils.binsToUse
    
    # for 1D bins - assumed ordered, non-overlapping
    
    binEdges = np.append([binList[i]['FeH'][0] for i in range(len(binList))], binList[-1]['FeH'][1])
    binMidpoints = (binEdges[:-1] + binEdges[1:])/2
    
    numberDM = densityModel
    
    
    
    
    
    
    
    
    
    
class densityModel:
    def __init__(self, binList):
        self.binList = binList
        self.logAmp = np.zeros(len(binList))
        self.aR = np.zeros(len(binList))
        self.az = np.zeros(len(binList))
        for i, binDict in enumerate(self.binList):
            self.logAmp[i], self.aR[i], self.az[i] = loadFitResults(binDict)
            
    def binIndex(**kwargs):
    """
    takes kwargs of values, finds bin those values lie in
    Assumes no overlapping bins
    """
    for i, bin in enumerate(self.binList):
        isIn = True
        for key, limits in bin.items():
            isIn &= (limits[0]<=kwargs[key]) & (kwargs[key]<limits[1])
        if isIn: return i
    return np.nan # not in any bin

    def distribution(paramSpacePoint, position, normalised=False):
        R, z = position
        i = self.binIndex(**paramSpacePoint)
        return np.exp(self.logAmp[i] - self.aR[i]*(R-myUtils.R_Sun) - self.az[i]*np.abs(z))
    
    def integrated(paramSpacePoints, normalised=False):)
        i = self.binIndex(**paramSpacePoint)
        return 4*np.pi*np.exp(self.logAmp[i] + self.aR[i]*myUtils.R_Sun)/(self.aR[i]**2 * self.az[i])


def loadFitResults(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'fit_results.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

def loadNRG2mass(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'NRG2mass.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

def binParamSpaceVol(binDict):
    size=1
    for limits in binDict.values():
        size*=(limits[1]-limits[0])
    return size
    
    
if __name__=='__main__':
    main()
