# Goal is unified, easy plotting of any combination of bins

import numpy as np
import os
import pickle
import git

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

import mySetup
import myIsochrones


cmap1 = mpl.colormaps['Blues']
cmap2 = mpl.colormaps['hsv']


repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:7]
print(repo)
print(sha)

plotDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/'



def main():
    # plotFeH()
    # plotMgFeFeH(False)
    # plotMgFeFeH(True)
    # plotEAGLE()
    # plotageFeH()
    # plot_scales_MgFeFeHvsFeH()
    # plot_data_MgFeFeHvsFeH()
    pass



class Galaxy:
    def __init__(self, FeHEdges, aFeEdges, amp, aR, az, sig_logNuSun, sig_aR, sig_az, data):
        """
        amp etc are arrays with [FeH index, aFe index]
        """
        self.FeHEdges = FeHEdges
        self.aFeEdges = aFeEdges
        self.amp = amp
        self.aR = aR
        self.az = az
        self.sig_logNuSun = sig_logNuSun # unsure if useful
        self.sig_aR = sig_aR
        self.sig_az = sig_az
        self.data = data
        
        self.shape = self.amp.shape
        self.FeHWidths = FeHEdges[1:] - FeHEdges[:-1]
        self.FeHMidpoints = (FeHEdges[1:] + FeHEdges[:-1])/2
        self.aFeWidths = aFeEdges[1:] - aFeEdges[:-1]
        self.aFeMidpoints = (aFeEdges[1:] + aFeEdges[:-1])/2
        self.vols = self.FeHWidths.reshape(-1,1) * self.aFeWidths
        assert self.vols.shape==self.amp.shape
        
    @classmethod
    def loadFromBins(cls, weightingNum, FeHEdges=mySetup.FeHEdges, aFeEdges=mySetup.aFeEdges):
        """
        
        """
        shape = (len(FeHEdges)-1, len(aFeEdges)-1)
        amp = np.zeros(shape)
        aR = np.zeros(shape)
        az = np.zeros(shape)
        sig_logNuSun = np.zeros(shape)
        sig_aR = np.zeros(shape)
        sig_az = np.zeros(shape)
        data = np.zeros((3, *shape))
        
        FeHWidths = FeHEdges[1:] - FeHEdges[:-1]
        aFeWidths = aFeEdges[1:] - aFeEdges[:-1]
        vols = FeHWidths.reshape(-1,1) * aFeWidths
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                binDir  = os.path.join(mySetup.dataDir, 'bins', f'FeH_{FeHEdges[i]}_{FeHEdges[i+1]}_aFe_{aFeEdges[j]}_{aFeEdges[i+1]}')
                
                isogrid = myIsochrones.loadGrid()
                NRG2SMM = myIsochrones.NRG2SMM(isogrid, weightingNum)
                
                with open(os.path.join(binDir, 'data.dat'), 'rb') as f0:
                    data[:,i,j] = np.array(pickle.load(f0))
                
                with open(os.path.join(binDir, 'noPyro_fit_results.dat'), 'rb') as f1:
                    logA, aR[i,j], az[i,j] = pickle.load(f1)
                    
                with open(os.path.join(binDir, 'noPyro_fit_sigmas.dat'), 'rb') as f1:
                    sig_logNuSun[i,j], sig_aR[i,j], sig_az[i,j] = pickle.load(f1)
                    
                    
                amp[i,j] = NRG2SMM*np.exp(logA)/vols
        return cls(FeHEdges, aFeEdges, amp, aR, az, sig_logNuSun, sig_aR, sig_az, data)
    
    
    def hist(self, R=mySetup.R_Sun, z=mySetup.z_Sun, normalised=False):
        hist = self.amp*np.exp( - self.aR*(R-mySetup.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.hist(R,z)) #assumes bins cover whole distribution
    
    def integratedHist(self, Rlim=None, zlim=None, normalised=False):
        """integrates R=0 to R and z=-z to z, with default of whole galaxy"""
        hist = np.where((self.aR>0)&(self.az>0), 4*np.pi*self.amp*np.exp(self.aR*mySetup.R_Sun)/(self.aR**2 * self.az), 0)
        # volume technically infinite for negative a, this only occurs when negligible stars in bin
        if Rlim!=None:
            hist *= (1 - (1+self.aR*Rlim)*np.exp(-self.aR*Rlim))
        if zlim!=None:
            hist *= (1 - np.exp(-self.az*zlim))
            
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.integratedHist())
        
    def zintegratedHist(self, R=mySetup.R_Sun, zlim=None, normalised=False):
        """integrates z=-z to z at given R, with default of whole vertical range"""
        hist = np.where((self.az>0), 2*self.amp*np.exp(-self.aR*(R-mySetup.R_Sun))/(self.az), 0)
        # volume technically infinite for negative a, this only occurs when negligible stars in bin
        if zlim!=None:
            hist *= (1 - np.exp(-self.az*zlim))
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.zintegratedHist())
        
        
    def FeH(self, name, hist, integrated=False, Rlim=None, zlim=None, normalised=False, prop2Z=True):
        """
        Makes Distribution object from hist
        """
        pass
        
    #     if integrated==False:
    #         #at point
    #         hist = self.hist(R, z, normalised)
    #         perVolume = True
    #     else:
    #         hist = self.integratedHist(Rlim, zlim, normalised)
    #         perVolume = False
        
    #     axes = list(range(len(self.widths)))
    #     axes.remove(FeHaxis)
    #     axes = tuple(axes)
    #     FeHhist = (hist * self.vols).sum(axis=axes)/FeHwidths
    #     print(FeHhist)
    #     return Distributions(FeHhist, FeHedges, name, self.binType, perVolume, n
    
    # def FeH(self, name, R=mySetup.R_Sun, z=mySetup.z_Sun, integrated=False, Rlim=None, zlim=None, normalised=False, prop2Z=True):
        
        
    #     if integrated==False:
    #         #at point
    #         hist = self.hist(R, z, normalised)
    #         perVolume = True
    #     else:
    #         hist = self.integratedHist(Rlim, zlim, normalised)
    #         perVolume = False
        
    #     axes = list(range(len(self.widths)))
    #     axes.remove(FeHaxis)
    #     axes = tuple(axes)
    #     FeHhist = (hist * self.vols).sum(axis=axes)/FeHwidths
    #     print(FeHhist)
    #     return Distributions(FeHhist, FeHedges, name, self.binType, perVolume, normalised, prop2Z)



class Distributions:
    """Contains the binned FeH distributions, and methods to get corresponding smoothed and fH2O distributions.
    Also contiains data needed for plotting for ease
    Should normaliser be at later stage?"""
    
    alpha = 1
    
    def __init__(self, name, FeHHist, FeHEdges=mySetup.FeHEdges, perVolume=True, normalised=False, powerIndex=1):
        self.FeHEdges = FeHEdges
        self.FeHWidths = self.FeHEdges[1:] - self.FeHEdges[:-1]
        self.FeHMidpoints = (self.FeHEdges[1:] + self.FeHEdges[:-1])/2
        self.name = name
        self.perVolume = perVolume
        if not normalised:
            self.FeHHist = FeHHist
        else:
            self.FeHHist = FeHHist/np.sum(self.FeHWidths*FeHHist)
        self.powerIndex = powerIndex
        
        y = np.append(0, np.cumsum(self.FeHWidths*self.FeHHist))
        dist = scipy.interpolate.CubicSpline(self.FeHEdges, y, bc_type='clamped', extrapolate=True).derivative()
        def FeHdistFunc(FeH):
            # sets dist to zero outside range
            return np.where((self.FeHEdges[0]<=FeH)&(FeH<self.FeHEdges[-1]), dist(FeH), 0)
        self.FeHdist = FeHdistFunc
        
        def ISOsPerFeH(FeH):
            return self.alpha*(10**(powerIndex*FeH))*self.FeHdist(FeH)
        
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
        # could this be cls() instead?
        return self.__class__(self.name, self.FeHHist, self.FeHEdges, self.perVolume, normalised=True, powerIndex=self.powerIndex)
        
    def plot(self, saveDir=plotDir):
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
        
        fig, ax = plt.subplots()
        ax.bar(self.FeHmidpoints, self.FeHhist, width = self.FeHwidths, alpha=0.5)
        ax.plot(FeHplotPoints, self.FeHdist(FeHplotPoints), color='C1')
        ax.vlines(-0.4, 0, self.FeHdist(0), color='C2', alpha=0.5)
        ax.vlines( 0.4, 0, self.FeHdist(0), color='C2', alpha=0.5)
        ax.set_xlabel(r'[Fe/H]')
        ax.set_ylabel(f'Stellar mass distribution ({FeHunit})')
        ax.set_title(self.name)
        path = os.path.join(saveDir, self.name + '_FeH' + '.pdf')
        fig.savefig(path, dpi=300)    
        
        fig, ax = plt.subplots()
        ax.plot(fH2OplotPoints, self.fH2Odist(fH2OplotPoints))
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$f_\mathrm{H_2O}$')
        ax.set_ylabel(f'ISO distribution ({fH2Ounit})')
        ax.set_title(self.name)
        path = os.path.join(saveDir, self.name + '_fH2O' + '.pdf')
        fig.savefig(path, dpi=300)    
        
        fig, ax = plt.subplots()
        ax.bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  self.counts)
        ax.set_ylabel(f'ISO distribution ({fH2OintUnit})')
        ax.set_title(self.name)
        path = os.path.join(saveDir, self.name + '_bar' + '.pdf')
        fig.savefig(path, dpi=300)
          
        
        

    def plotWith(self, dists2, saveDir=plotDir):
        
        dists1 = self.butNormalised()
        dists2 = dists2.butNormalised()
        
        os.makedirs(saveDir, exist_ok=True)
        
        FeHunit=''
        fH2Ounit=''
        fH2OintUnit='' #sort later
        
        FeHplotPoints = np.linspace(min(dists1.FeHedges[0], dists2.FeHedges[0]),
                                    max(dists1.FeHedges[-1], dists2.FeHedges[-1]),
                                    10*max(len(dists1.FeHwidths), len(dists2.FeHwidths)))
        fH2OplotPoints = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
        
        names = f'{self.name}+{dists2.name}'
        
        fig, ax = plt.subplots()
        ax.plot(FeHplotPoints, dists1.FeHdist(FeHplotPoints), label=self.name)
        ax.plot(FeHplotPoints, dists2.FeHdist(FeHplotPoints), label=dists2.name)
        ax.vlines(-0.4, 0, dists1.FeHdist(0), color='C2', alpha=0.5)
        ax.vlines( 0.4, 0, dists1.FeHdist(0), color='C2', alpha=0.5)
        ax.legend()
        ax.set_xlabel(r'[Fe/H]')
        ax.set_ylabel(f'Stellar mass distribution ({FeHunit})')
        ax.set_title(names)
        path = os.path.join(saveDir, names + '_FeH' + '.pdf')
        fig.savefig(path, dpi=300)
        
        ax.plot(fH2OplotPoints, dists1.fH2Odist(fH2OplotPoints), label=self.name)
        ax.plot(fH2OplotPoints, dists2.fH2Odist(fH2OplotPoints), label=dists2.name)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.set_xlabel(r'$f_\mathrm{H_2O}$')
        ax.set_ylabel(f'ISO distribution ({fH2Ounit})')
        ax.set_title(names)
        path = os.path.join(saveDir, names + '_fH2O' + '.pdf')
        fig.savefig(path, dpi=300)
        
        ax.bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  dists1.counts, alpha=0.5, label=dists1.name)
        ax.bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  dists2.counts, alpha=0.5, label=dists2.name)
        ax.legend()
        ax.set_title(names)
        path = os.path.join(saveDir, names + '_bar' + '.pdf')
        fig.savefig(path, dpi=300)
        
                    
        
    

# Defining comp:
FeH_p = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
compPoly = np.polynomial.polynomial.Polynomial.fit(FeH_p, fH2O_p, 3)
FeHlow = FeH_p[0]
FeHhigh = FeH_p[-1]
fH2Olow = compPoly(FeH_p[-1])
fH2Ohigh = compPoly(FeH_p[0])
def comp(FeH):
    return np.where(FeHlow<=FeH, np.where(FeH<FeHhigh,
                                          compPoly(FeH),
                                          fH2Olow), fH2Ohigh)
    
def compInv(fH2O):
    """inv may not work with array inputs"""
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
    
    
    
    
if __name__=='__main__':
    pass
    
    
    
    
    
    