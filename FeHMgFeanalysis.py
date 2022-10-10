import os
from functools import partial
import pickle

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import quad
import matplotlib.pyplot as plt
import git

import myUtils

# for 1D bins - assumed ordered, non-overlapping, equal size

POLYDEG=3

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:7]
print(repo)
print(sha)


saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/FeHMgFeanalysis/{POLYDEG}/'

binsDir = '/Users/hopkinsm/data/APOGEE/bins/'

FeHedges = myUtils._FeH_edges_for_MgFe
MgFeedges = myUtils._MgFe_edges

def main():
    
    
    binList = [{'FeH': (FeHedges[bindex(i)[1]], FeHedges[bindex(i)[1]+1]), 'MgFe': (MgFeedges[bindex(i)[0]], MgFeedges[bindex(i)[0]+1])}
                                                         for i in range((len(MgFeedges)-1)*(len(FeHedges)-1))]
    
    
    FeHmidpoints = (FeHedges[:-1] + FeHedges[1:])/2
    FeHwidths = FeHedges[1:] - FeHedges[:-1]
    
    MgFemidpoints = (MgFeedges[:-1] + MgFeedges[1:])/2
    MgFewidths = MgFeedges[1:] - MgFeedges[:-1]
    
    logA     = np.zeros(shape)
    print(logA)
    aR       = np.zeros(shape)
    az       = np.zeros(shape)
    NRG2mass = np.zeros(shape)
    
    for i, binDict in enumerate(binList):
        bi = bindex(i)
        logA[bi], aR[bi], az[bi] = loadFitResults(binDict)
        if (logA[bi]==-999):
            # no stars in bin, fit not attempted
            aR[bi], az[bi] = 0.000001, 0.0000001
            NRG2mass[bi] = 0
        else:
            NRG2mass[bi] = loadNRG2mass(binDict)
            
    print(np.meshgrid(FeHmidpoints,MgFemidpoints))
            
    widthmgs = np.meshgrid(FeHwidths, MgFewidths)
    binAreas = widthmgs[0]*widthmgs[1]
    print(binAreas)
        
    #NRGDM = densityModel(FeHEdges, np.exp(FeHnumberlogA)/FeHWidths, aR, az)
    StellarMassDM = densityModel(FeHedges, NRG2mass*np.exp(logA)/(binAreas), aR, az)
    
    print('Check hist')
    print(StellarMassDM.integratedHist())
    
    fig, axs = plt.subplots(ncols=3, figsize=[12, 4])
    axs[0].imshow(StellarMassDM.integratedHist(), origin='lower', extent=(FeHedges[0], FeHedges[-1], MgFeedges[0], MgFeedges[-1]))
    axs[1].imshow(aR[StellarMassDM.integratedHist()>1e8], origin='lower', extent=(FeHedges[0], FeHedges[-1], MgFeedges[0], MgFeedges[-1]))
    axs[2].imshow(az, origin='lower', extent=(FeHedges[0], FeHedges[-1], MgFeedges[0], MgFeedges[-1]))


shape = (len(MgFeedges)-1, len(FeHedges)-1)

def bindex(indx):
    "ravels or unravels index, depending on input"
    dims = shape
    if type(indx) is tuple:
        return np.ravel_multi_index(indx,dims)
    else:
        return np.unravel_index(indx,dims)



def saveFig(fig, name):
    os.makedirs(saveDir, exist_ok=True)
    path = saveDir+name
    fig.savefig(path, dpi=300)

# Defining comp:
FeH_p = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
compPoly = np.polynomial.polynomial.Polynomial.fit(FeH_p, fH2O_p, POLYDEG)
FeHlow = FeH_p[0]
FeHhigh = FeH_p[-1]
fH2Olow = compPoly(FeH_p[-1])
fH2Ohigh = compPoly(FeH_p[0])

def comp(FeH):
    return np.where(FeHlow<=FeH, np.where(FeH<FeHhigh,
                                          compPoly(FeH),
                                          fH2Olow), fH2Ohigh)
        
def compInv(fH2O):
    """inv mayn't work with array inputs"""
    if np.ndim(fH2O)==0:
        val = fH2O
        if fH2Olow<=val<fH2Ohigh:
            allroots = (compPoly-val).roots()
            myroot = allroots[(FeH_p[0]<=allroots)&(allroots<FeH_p[-1])]
            assert len(myroot)==1 # checks not multiple roots
            assert np.isreal(myroot[0])
            return np.real(myroot[0])
        else:
            return np.nan
    else:
        returnArray = np.zeros_like(fH2O)
        for i, val in enumerate(fH2O):
            if fH2Olow<=val<fH2Ohigh:
                allroots = (compPoly-val).roots()
                myroot = allroots[(FeH_p[0]<=allroots)&(allroots<FeH_p[-1])]
                assert len(myroot)==1 # checks not multiple roots
                assert np.isreal(myroot[0])
                returnArray[i] = np.real(myroot[0])
            else:
                returnArray[i] =  np.nan
        return returnArray
        
def compDeriv(FeH):
    return np.where((FeHlow<=FeH)&(FeH<FeHhigh), compPoly.deriv()(FeH), 0)
            

for x in [-0.4+0.0001, -0.2, 0, 0.2, 0.4-0.0001]:
    assert np.isclose(compInv(comp(x)), x) #checks inverse works


def loadFitResults(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'fit_results.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

def loadNRG2mass(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'NRG2mass.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)


def SM2ISO(FeHdist, alpha=1, normalised=False):
    def integrand(FeH):
        return (10**FeH)*FeHdist(FeH)
    normFactor = alpha*quad(integrand, -3, 3, limit=200)[0] if normalised else 1
    #ISOdist = lambda fH2O: -alpha*FeHdist(compInv(fH2O))/(normFactor*compDeriv(compInv(fH2O)))
    def ISOdist(fH2O):
        return -alpha*(10**compInv(fH2O))*FeHdist(compInv(fH2O))/(normFactor*compDeriv(compInv(fH2O)))
    lowerEndCount = alpha*quad(integrand, FeHhigh, 3, limit=200)[0]/normFactor
    upperEndCount = alpha*quad(integrand, -3, FeHlow, limit=200)[0]/normFactor
    return (ISOdist, lowerEndCount, upperEndCount)

    
def hist2dist(edges, hist, normalised=False):
    widths = edges[1:] - edges[:-1]
    if not normalised:
        y = np.append(0, np.cumsum(widths*hist))
    else:
        y = np.append(0, np.cumsum(widths*hist)/np.sum(widths*hist))
    
    dist = CubicSpline(edges, y, bc_type='clamped', extrapolate=True).derivative()
    # dist = PchipInterpolator(edges, y).derivative() # gives correct boundary conditions (ie flat) if at least one empty bin on either end
    def distFunc(FeH):
        return np.where((edges[0]<=FeH)&(FeH<edges[-1]), dist(FeH), 0)
    return distFunc

    
    
class densityModel:
    """distribution of some 'quantity' (number or mass) in volume
    and FeH"""
    def __init__(self, edges, distAmp, aR, az):
        """amp should be"""
        self.edges = edges
        self.widths = edges[1:]-edges[:-1]
        self.midpoints = (edges[:-1]+edges[1:])/2
        self.distAmp = distAmp
        self.aR = aR
        self.az = az

    def hist(self, position=(myUtils.R_Sun, myUtils.z_Sun), normalised=False):
        R, z = position
        dist = self.distAmp*np.exp( - self.aR*(R-myUtils.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return dist
        else:
            return dist/sum(self.widths*self.hist(position))
    
    def integratedHist(self, normalised=False):
        dist = 4*np.pi*self.distAmp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az)
        if not normalised:
            return dist
        else:
            return dist/sum(self.widths*self.integratedHist())
        
    def histWithin(self, R, z=np.inf):
        return (4*np.pi*self.distAmp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az))*(1-np.exp(-self.az*z))*(1 - (1+self.aR*R)*np.exp(-self.aR*R))
        

    
if __name__=='__main__':
    main()
