
# Goal is unified, easy plotting of any combination of bins

import numpy as np
import os
import pickle

import scipy

import myUtils




POLYDEG = 3

def plotFeH():
    
    bins=0
    
    galaxy = Galaxy.loadFromBins()
    
    
def plotMgFeFeH():
    pass
    
    
    
    
    
    


class Galaxy:
    def __init__(self, labels, edges, amp, aR, az):
        """
        edges[i] is array of edges of ith dimention
        amp, aR and az each have element corresponding to bin
        """
        self.labels = labels
        self.edges = edges
        self.amp = amp
        self.aR = aR
        self.az = az
        
        self.shape = self.amp.shape
        self.widths = [(edges[i][1:] - edges[i][:-1]) for i in range(len(self.shape))]
        self.midpoints = [(edges[i][1:] + edges[i][:-1])/2 for i in range(len(self.shape))]
        self.vols = np.zeros(self.shape)
        for binNum in range(np.prod(self.shape)):
            multiIndex = np.unravel_index(binNum, self.shape)
            self.vols[multiIndex] = np.prod([self.widths[i][multiIndex[i]] for i in range(len(self.shape))])
        
    @classmethod
    def loadFromBins(cls, labels, edges):
        """
        edges[i] is list of np.array edges of ith dimention
        labels[i] should be string of ith quantity
         """
        sortIndices = np.argsort(labels)
        labels = [labels[si] for si in sortIndices]
        edges = [edges[si] for si in sortIndices] # ensures labels are sorted for consistency
        
        shape = tuple((len(edges[i])-1) for i in range(len(labels)))
        amp = np.zeros(shape)
        aR = np.zeros(shape)
        az = np.zeros(shape)
        
        for binNum in range(amp.size):
            multiIndex = np.unravel_index(binNum, shape)
            limits = np.array([[edges[i][multiIndex[i]], edges[i][multiIndex[i]+1]] for i in range(len(labels))])
            print(limits)
            binDir  = os.path.join(myUtils.localDataDir, 'bins', binName(labels, limits))
            
            with open(os.path.join(binDir, 'fit_results.dat'), 'rb') as f1:
                logA, aR[multiIndex], az[multiIndex] = pickle.load(f1)
            with open(os.path.join(binDir, 'NRG2mass.dat'), 'rb') as f2:
                NRG2Mass = pickle.load(f2)
            amp[multiIndex] = NRG2Mass*np.exp(logA)/(np.prod(limits[:,1]-limits[:,0]))
        return cls(labels, edges, amp, aR, az)

    def hist(self, R=myUtils.R_Sun, z=myUtils.z_Sun, normalised=False):
        hist = self.amp*np.exp( - self.aR*(R-myUtils.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.hist(R,z)) #assumes bins cover whole distribution
    
    def integratedHist(self, lims=None, normalised=False):
        """integrates R=0 to R and z=-z to z, with default of whole galaxy"""
        hist = 4*np.pi*self.amp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az)
        if lims is not None:
            R, z = lims
            hist *= (1 - (1+self.aR*R)*np.exp(-self.aR*R)) * (1 - np.exp(-self.az*z))
            
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.integratedHist())
        
    def FeH(self, R=myUtils.R_Sun, z=myUtils.z_Sun, normalised=False):
        return (self.hist(R,z,normalised)*self.MgFeWidths[:,np.newaxis]).sum(axis=0)



class binnedFeHDist:
    def __init__(self, hist, edges, name, perVolume):
        self.hist = hist
        self.edges = edges
        self.name = name
        self.perVolume = perVolume


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
        
def SM2ISO(FeHdist, alpha=1, normalised=False):
    def integrand(FeH):
        return (10**FeH)*FeHdist(FeH)
    
    normFactor = alpha*scipy.integrate.quad(integrand, -3, 3, limit=200)[0] if normalised else 1
    
    def ISOdist(fH2O):
        return -alpha*(10**compInv(fH2O))*FeHdist(compInv(fH2O))/(normFactor*compDeriv(compInv(fH2O)))
    
    lowerEndCount = alpha*scipy.integrate.quad(integrand, FeHhigh, 3, limit=200)[0]/normFactor
    upperEndCount = alpha*scipy.integrate.quad(integrand, -3, FeHlow, limit=200)[0]/normFactor
    return (ISOdist, lowerEndCount, upperEndCount)

def hist2dist(edges, hist, normalised=False):
    widths = edges[1:] - edges[:-1]
    if not normalised:
        y = np.append(0, np.cumsum(widths*hist))
    else:
        y = np.append(0, np.cumsum(widths*hist)/np.sum(widths*hist))
    
    dist = scipy.interpolate.CubicSpline(edges, y, bc_type='clamped', extrapolate=True).derivative()
    def distFunc(FeH):
        return np.where((edges[0]<=FeH)&(FeH<edges[-1]), dist(FeH), 0)
    return distFunc


def binName(labels, limits):
    return '_'.join(['_'.join([labels[i], f'{limits[i][0]:.3f}', f'{limits[i][1]:.3f}']) for i in range(len(limits))])


    
if __name__=='__main__':
    plotFeH()