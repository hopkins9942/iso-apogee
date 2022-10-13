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






def main():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:7]
    print(repo)
    print(sha)
    
    FeHEdges = myUtils._FeH_edges_for_MgFe
    MgFeEdges = myUtils._MgFe_edges
    
    FeHMidpoints = (FeHEdges[:-1] + FeHEdges[1:])/2
    FeHWidths = FeHEdges[1:] - FeHEdges[:-1]
    
    MgFeMidpoints = (MgFeEdges[:-1] + MgFeEdges[1:])/2
    MgFeWidths = MgFeEdges[1:] - MgFeEdges[:-1]
    
            
    binAreas = MgFeWidths[:,np.newaxis]*FeHWidths
    
    StellarMassDM = FeHMgFeModel.loadFromBins(FeHEdges, MgFeEdges)
    
    fig, axs = plt.subplots(ncols=5, figsize=[16, 4])
    titles = ['density at sun', 'integrated density', 'thick disk', 'aR', 'az']
    for i, im in enumerate([StellarMassDM.hist(), StellarMassDM.integratedHist(), StellarMassDM.hist(R=4,z=1), StellarMassDM.aR, StellarMassDM.az]):
        axs[i].imshow(im, vmin=0, origin='lower', aspect='auto', extent=(FeHEdges[0], FeHEdges[-1], MgFeEdges[0], MgFeEdges[-1]))
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('[Fe/H]')
        axs[i].set_ylabel('[alpha/Fe]')
    fig.set_tight_layout(True)
    saveFig(fig, 'Bovy2012')
    
    print(StellarMassDM.hist(R=4,z=1))
    
    
    FeHHist = StellarMassDM.FeHHist()
    FeHDist = hist2dist(FeHEdges, FeHHist)
    
    FeHplotPoints = np.linspace(FeHEdges[0], FeHEdges[-1], 10*len(FeHWidths))
    fH2OplotPoints = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
    fH2Odist, lowerCount, upperCount = SM2ISO(FeHDist)
    middleCount = quad(fH2Odist, fH2OplotPoints[0], fH2OplotPoints[-1])[0]

    
    fig, axs = plt.subplots(ncols=3, figsize=[12, 4])
    axs[0].bar(FeHMidpoints, FeHHist, width = FeHWidths, alpha=0.5)
    axs[0].plot(FeHplotPoints, FeHDist(FeHplotPoints), color='C1')
    axs[0].plot([-0.4,-0.4], [0,FeHDist(0)], color='C2', alpha=0.5)
    axs[0].plot([ 0.4, 0.4], [0,FeHDist(0)], color='C2', alpha=0.5)
    axs[0].set_xlabel(r'[Fe/H]')
    axs[0].set_ylabel(f'Stellar mass distribution')

    
    # print(f'Check spline method: {plotNum}')
    # print(np.array([quad(FeHdist, FeHedges[i], FeHedges[i+1])[0]/FeHwidths[i]
    #                     for i in range(len(FeHmidpoints))]) - FeHhist)
    

    axs[1].plot(fH2OplotPoints, fH2Odist(fH2OplotPoints))
    axs[1].set_ylim(bottom=0)
    axs[1].set_xlabel(r'$f_\mathrm{H_2O}$')
    axs[1].set_ylabel(f'ISO distribution)')
    axs[2].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
              [lowerCount, middleCount, upperCount])
    axs[2].set_ylabel(f'ISO distribution')
    fig.suptitle(f'local')
    fig.set_tight_layout(True)
    saveFig(fig, f'localISO.png')



   

# shape = (len(MgFeedges)-1, len(FeHedges)-1)

# def bindex(indx):
#     "ravels or unravels index, depending on input"
#     dims = shape
#     if type(indx) is tuple:
#         return np.ravel_multi_index(indx,dims)
#     else:
#         return np.unravel_index(indx,dims)



def saveFig(fig, name):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:7]
    saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/FeHMgFeanalysis/{POLYDEG}/'
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


# def loadFitResults(binDict):
#     path = os.path.join(binsDir, myUtils.binName(binDict), 'fit_results.dat')
#     with open(path, 'rb') as f:
#         return pickle.load(f)

# def loadNRG2mass(binDict):
#     path = os.path.join(binsDir, myUtils.binName(binDict), 'NRG2mass.dat')
#     with open(path, 'rb') as f:
#         return pickle.load(f)


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

    
    
# class densityModel:
#     """distribution of some 'quantity' (number or mass) in volume
#     and FeH"""
#     def __init__(self, edges, distAmp, aR, az):
#         """amp should be"""
#         self.edges = edges
#         self.widths = edges[1:]-edges[:-1]
#         self.midpoints = (edges[:-1]+edges[1:])/2
#         self.distAmp = distAmp
#         self.aR = aR
#         self.az = az

#     def hist(self, position=(myUtils.R_Sun, myUtils.z_Sun), normalised=False):
#         R, z = position
#         dist = self.distAmp*np.exp( - self.aR*(R-myUtils.R_Sun) - self.az*np.abs(z))
#         if not normalised:
#             return dist
#         else:
#             return dist/sum(self.widths*self.hist(position))
    
#     def integratedHist(self, normalised=False):
#         dist = 4*np.pi*self.distAmp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az)
#         if not normalised:
#             return dist
#         else:
#             return dist/sum(self.widths*self.integratedHist())
        
#     def histWithin(self, R, z=np.inf):
#         return (4*np.pi*self.distAmp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az))*(1-np.exp(-self.az*z))*(1 - (1+self.aR*R)*np.exp(-self.aR*R))
        
    
class FeHMgFeModel:
    """made 20221012"""
    def __init__(self, FeHEdges, MgFeEdges, amp, aR, az):
        self.MgFeEdges = MgFeEdges
        self.FeHEdges = FeHEdges
        self.FeHWidths = FeHEdges[1:] - FeHEdges[:-1]
        self.MgFeWidths = MgFeEdges[1:] - MgFeEdges[:-1]
        self.areas = self.MgFeWidths[:,np.newaxis]*self.FeHWidths
        self.amp = amp
        self.aR = aR
        self.az = az
    
    @classmethod
    def loadFromBins(cls, FeHEdges, MgFeEdges):
        NMgFe = len(MgFeEdges)-1
        NFeH = len(FeHEdges)-1
        amp = np.zeros((NMgFe, NFeH))
        aR = np.zeros((NMgFe, NFeH))
        az = np.zeros((NMgFe, NFeH))
        for i in range(NMgFe):
            for j in range(NFeH):
                binsDir = '/Users/hopkinsm/data/APOGEE/bins/'
                ijBinDir = os.path.join(binsDir, f'FeH_{FeHEdges[j]:.3f}_{FeHEdges[j+1]:.3f}_MgFe_{MgFeEdges[i]:.3f}_{MgFeEdges[i+1]:.3f}')
                resPath  = os.path.join(ijBinDir, 'fit_results.dat')
                NRG2MPath= os.path.join(ijBinDir, 'NRG2mass.dat')
                with open(resPath, 'rb') as f1:
                    logA, aR[i,j], az[i,j] = pickle.load(f1)
                with open(NRG2MPath, 'rb') as f2:
                    NRG2Mass = pickle.load(f2)
                amp[i,j] = NRG2Mass*np.exp(logA)/((MgFeEdges[i+1]-MgFeEdges[i])*(FeHEdges[j+1]-FeHEdges[j]))
        return cls(FeHEdges, MgFeEdges,  amp, aR, az)
    
    def hist(self, R=myUtils.R_Sun, z=myUtils.z_Sun, normalised=False):
        hist = self.amp*np.exp( - self.aR*(R-myUtils.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return hist
        else:
            return hist/sum(self.areas*self.hist(R,z)) #assumes bins cover whole distribution
    
    def integratedHist(self, normalised=False):
        hist = 4*np.pi*self.amp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az)
        if not normalised:
            return hist
        else:
            return hist/sum(self.areas*self.integratedHist())
        
    def FeHHist(self, R=myUtils.R_Sun, z=myUtils.z_Sun, normalised=False):
        return (self.hist(R,z,normalised)*self.MgFeWidths[:,np.newaxis]).sum(axis=0)
        
        


    
if __name__=='__main__':
    main()


