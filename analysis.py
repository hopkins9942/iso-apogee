# Goal is unified, easy plotting of any combination of bins

import numpy as np
import os
import pickle
import git

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

import myUtils


cmap1 = mpl.colormaps['Blues']
cmap2 = mpl.colormaps['hsv']


repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:7]
print(repo)
print(sha)



def main():
    # plotFeH()
    # plotMgFeFeH(False)
    # plotMgFeFeH(True)
    # plotEAGLE()
    # plotageFeH()
    # plot_scales_MgFeFeHvsFeH()
    # plot_data_MgFeFeHvsFeH()
    



class Galaxy:
    def __init__(self, labels, edges, amp, aR, az, sig_logNuSun, sig_aR, sig_az, data):
        """
        edges[i] is array of edges of ith dimention
        amp, aR and az each have element corresponding to bin
        """
        sortIndices = np.argsort(labels)
        self.labels = [labels[si] for si in sortIndices]
        self.binType = ''.join(self.labels)
        self.edges = [edges[si] for si in sortIndices] # ensures labels are sorted for consistency
        self.amp = amp
        self.aR = aR
        self.az = az
        #self.over100 = over100
        self.sig_logNuSun = sig_logNuSun # unsure if useful
        self.sig_aR = sig_aR
        self.sig_az = sig_az
        self.data = data
        
        self.shape = self.amp.shape
        self.widths = [(self.edges[i][1:] - self.edges[i][:-1]) for i in range(len(self.shape))]
        self.midpoints = [(self.edges[i][1:] + self.edges[i][:-1])/2 for i in range(len(self.shape))]
        self.vols = np.zeros(self.shape)
        for binNum in range(np.prod(self.shape)):
            multiIndex = np.unravel_index(binNum, self.shape)
            self.vols[multiIndex] = np.prod([self.widths[i][multiIndex[i]] for i in range(len(self.shape))])
        
    @classmethod
    def loadFromBins(cls, labels, edges, noPyro=True):
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
        sig_logNuSun = np.zeros(shape)
        sig_aR = np.zeros(shape)
        sig_az = np.zeros(shape)
        data = np.zeros((3, *shape))
        
        for binNum in range(amp.size):
            multiIndex = np.unravel_index(binNum, shape)
            limits = np.array([[edges[i][multiIndex[i]], edges[i][multiIndex[i]+1]] for i in range(len(labels))])
            binDir  = os.path.join(myUtils.localDataDir, 'bins', binName(labels, limits))
            with open(os.path.join(binDir, 'data.dat'), 'rb') as f0:
                data[0][multiIndex], data[1][multiIndex], data[2][multiIndex] = pickle.load(f0)
            
            if noPyro:
                with open(os.path.join(binDir, 'noPyro_fit_results.dat'), 'rb') as f1:
                    logA, aR[multiIndex], az[multiIndex] = pickle.load(f1)
                with open(os.path.join(binDir, 'noPyro_fit_sigmas.dat'), 'rb') as f1:
                    sig_logNuSun[multiIndex], sig_aR[multiIndex], sig_az[multiIndex] = pickle.load(f1)
            else:
                with open(os.path.join(binDir, 'fit_results.dat'), 'rb') as f1:
                    logA, aR[multiIndex], az[multiIndex] = pickle.load(f1)
                
            with open(os.path.join(binDir, 'NRG2mass.dat'), 'rb') as f2:
                NRG2Mass = pickle.load(f2)
                
            amp[multiIndex] = NRG2Mass*np.exp(logA)/(np.prod(limits[:,1]-limits[:,0]))
        return cls(labels, edges, amp, aR, az, sig_logNuSun, sig_aR, sig_az, data)
    
    
    def hist(self, R=myUtils.R_Sun, z=myUtils.z_Sun, normalised=False):
        hist = self.amp*np.exp( - self.aR*(R-myUtils.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.hist(R,z)) #assumes bins cover whole distribution
    
    def integratedHist(self, Rlim=None, zlim=None, normalised=False):
        """integrates R=0 to R and z=-z to z, with default of whole galaxy"""
        hist = np.where((self.aR>0)&(self.az>0), 4*np.pi*self.amp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az), 0)
        # volume technically infinite for negative a, this only occurs when negligible stars in bin
        if Rlim!=None:
            hist *= (1 - (1+self.aR*Rlim)*np.exp(-self.aR*Rlim))
        if zlim!=None:
            hist *= (1 - np.exp(-self.az*zlim))
            
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.integratedHist())
        
    def FeH(self, name, R=myUtils.R_Sun, z=myUtils.z_Sun, integrated=False, Rlim=None, zlim=None, normalised=False, prop2Z=True):
        FeHaxis = self.labels.index('FeH')
        FeHwidths = self.widths[FeHaxis]
        FeHedges = self.edges[FeHaxis]
        
        if integrated==False:
            #at point
            hist = self.hist(R, z, normalised)
            perVolume = True
        else:
            hist = self.integratedHist(Rlim, zlim, normalised)
            perVolume = False
        
        axes = list(range(len(self.widths)))
        axes.remove(FeHaxis)
        axes = tuple(axes)
        FeHhist = (hist * self.vols).sum(axis=axes)/FeHwidths
        print(FeHhist)
        return Distributions(FeHhist, FeHedges, name, self.binType, perVolume, normalised, prop2Z)



class Distributions:
    """Contains the binned FeH distributions, and methods to get corresponding smoothed and fH2O distributions.
    Also contiains data needed for plotting for ease
    Should normaliser be at later stage?"""
    
    alpha = 1
    
    def __init__(self, FeHhist, FeHedges, name, binType, perVolume, normalised=False, prop2Z=True):
        self.FeHedges = FeHedges
        self.FeHwidths = self.FeHedges[1:] - self.FeHedges[:-1]
        self.FeHmidpoints = (self.FeHedges[1:] + self.FeHedges[:-1])/2
        self.name = name
        self.binType = binType
        self.perVolume = perVolume
        if not normalised:
            self.FeHhist = FeHhist
        else:
            self.FeHhist = FeHhist/np.sum(self.FeHwidths*FeHhist)
        self.prop2Z = prop2Z
        
        y = np.append(0, np.cumsum(self.FeHwidths*self.FeHhist))
        dist = scipy.interpolate.CubicSpline(self.FeHedges, y, bc_type='clamped', extrapolate=True).derivative()
        def FeHdistFunc(FeH):
            return np.where((self.FeHedges[0]<=FeH)&(FeH<self.FeHedges[-1]), dist(FeH), 0)
        self.FeHdist = FeHdistFunc
        
        def ISOsPerFeH(FeH):
            if prop2Z:
                return self.alpha*(10**FeH)*self.FeHdist(FeH)
            else:
                return self.alpha*self.FeHdist(FeH) #check this
        
        lowerCount = scipy.integrate.quad(ISOsPerFeH, FeHhigh, 3, limit=200)[0]
        middleCount = scipy.integrate.quad(ISOsPerFeH, FeHlow, FeHhigh, limit=200)[0]
        upperCount = scipy.integrate.quad(ISOsPerFeH, -3, FeHlow, limit=200)[0]
        normFactor = (lowerCount+middleCount+upperCount) if normalised else 1
        
        def fH2OdistFunc(fH2O):
            return ISOsPerFeH(compInv(fH2O))/(normFactor*np.abs(compDeriv(compInv(fH2O))))
        
        self.fH2Odist = fH2OdistFunc
        self.counts = (lowerCount/normFactor, middleCount/normFactor, upperCount/normFactor)
        
    def butNormalised(self):
        """returns a seperate Distributions with normalised=True"""
        return self.__class__(self.FeHhist, self.FeHedges, self.name, self. binType, self.perVolume, normalised=True, prop2Z=self.prop2Z)
        
    def plot(self, path):
        saveDir = path #f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/analysis_{self.binType}_POLYDEG{POLYDEG}/'
        os.makedirs(saveDir, exist_ok=True)
        
        
        # FeHunit = (r'$\mathrm{M}_\odot \mathrm{dex}^{-1} \mathrm{kpc}^{-3}$' if self.perVolume
        #            else r'$\mathrm{M}_\odot \mathrm{dex}^{-1}$')
        # fH2Ounit = (r'$\mathrm{ISOs} \; \mathrm{kpc}^{-3}$' if self.perVolume
        #             else r'$\mathrm{ISOs}$')
        # fH2OintUnit = (r'$\mathrm{ISOs} \; \mathrm{kpc}^{-3}$' if self.perVolume
        #             else r'$\mathrm{ISOs}$')
        
        # if not self.normalised:
        #     FeHunit = r'$\mathrm{M}_\odot \mathrm{dex}^{-1}'
        
        #     fH2Ounit = (r'$\text{ISO } \;'
        
        FeHunit=''
        fH2Ounit=''
        fH2OintUnit='' #sort later
        
        FeHplotPoints = np.linspace(self.FeHedges[0], self.FeHedges[-1], 10*len(self.FeHwidths))
        fH2OplotPoints = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
        
        fig, axs = plt.subplots(ncols=3, figsize=[12, 4])
        axs[0].bar(self.FeHmidpoints, self.FeHhist, width = self.FeHwidths, alpha=0.5)
        axs[0].plot(FeHplotPoints, self.FeHdist(FeHplotPoints), color='C1')
        axs[0].vlines(-0.4, 0, self.FeHdist(0), color='C2', alpha=0.5)
        axs[0].vlines( 0.4, 0, self.FeHdist(0), color='C2', alpha=0.5)
        axs[0].set_xlabel(r'[Fe/H]')
        axs[0].set_ylabel(f'Stellar mass distribution ({FeHunit})')
        
        axs[1].plot(fH2OplotPoints, self.fH2Odist(fH2OplotPoints))
        axs[1].set_ylim(bottom=0)
        axs[1].set_xlabel(r'$f_\mathrm{H_2O}$')
        axs[1].set_ylabel(f'ISO distribution ({fH2Ounit})')
        
        axs[2].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  self.counts)
        axs[2].set_ylabel(f'ISO distribution ({fH2OintUnit})')
        
        fig.suptitle(self.name)
        fig.set_tight_layout(True)
        path = os.path.join(saveDir, self.name + '.pdf')
        fig.savefig(path, dpi=300)    
        
        

    def plotWith(self, dists2, path):
        assert self.binType==dists2.binType
        
        dists1 = self.butNormalised()
        dists2 = dists2.butNormalised()
        
        saveDir = path #f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/analysis_{self.binType}_POLYDEG{POLYDEG}/'
        os.makedirs(saveDir, exist_ok=True)
        
        FeHunit=''
        fH2Ounit=''
        fH2OintUnit='' #sort later
        
        FeHplotPoints = np.linspace(min(dists1.FeHedges[0], dists2.FeHedges[0]),
                                    max(dists1.FeHedges[-1], dists2.FeHedges[-1]),
                                    10*max(len(dists1.FeHwidths), len(dists2.FeHwidths)))
        fH2OplotPoints = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
        
        fig, axs = plt.subplots(ncols=3, figsize=[12, 4])
        axs[0].plot(FeHplotPoints, dists1.FeHdist(FeHplotPoints), label=self.name)
        axs[0].plot(FeHplotPoints, dists2.FeHdist(FeHplotPoints), label=dists2.name)
        axs[0].vlines(-0.4, 0, dists1.FeHdist(0), color='C2', alpha=0.5)
        axs[0].vlines( 0.4, 0, dists1.FeHdist(0), color='C2', alpha=0.5)
        axs[0].legend()
        axs[0].set_xlabel(r'[Fe/H]')
        axs[0].set_ylabel(f'Stellar mass distribution ({FeHunit})')
        
        axs[1].plot(fH2OplotPoints, dists1.fH2Odist(fH2OplotPoints), label=self.name)
        axs[1].plot(fH2OplotPoints, dists2.fH2Odist(fH2OplotPoints), label=dists2.name)
        axs[1].set_ylim(bottom=0)
        axs[1].legend()
        axs[1].set_xlabel(r'$f_\mathrm{H_2O}$')
        axs[1].set_ylabel(f'ISO distribution ({fH2Ounit})')
        
        axs[2].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  dists1.counts, alpha=0.5, label=dists1.name)
        axs[2].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  dists2.counts, alpha=0.5, label=dists2.name)
        axs[2].legend()
        
        figname = f'{self.name} vs {dists2.name}.pdf'
        fig.suptitle(figname)
        fig.set_tight_layout(True)
        path = os.path.join(saveDir, figname)
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