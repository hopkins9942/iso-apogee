# Goal is unified, easy plotting of any combination of bins
import time 

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

plt.rcParams.update({
    "text.usetex": True})


repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:7]
print(repo)
print(sha)

plotDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/'




# really should sort integral warnings

def main():
    
    
    #for paper:
    makePlots(0,0,0)
    for wn in range(3):
        Zindex=1 # num proportional to Z metallicity
        makePlots(wn,wn,Zindex)
        
    #tests
    # G = Galaxy.loadFromBins(ESFweightingNum=0, NRG2SMMweightingNum=0)
    # R = np.linspace(0,20,11)
    # R = (R[:-1]+R[1:])/2
    
    # for r in R:
    #     print(r)
    #     D = Distributions(f'r={r}', G.FeH(G.zintegratedHist(r)), normalised=True, ISONumZIndex=1)
    #     D.plot()
        
    
    
    # old:
    # plotFeH()
    # plotMgFeFeH(False)
    # plotMgFeFeH(True)
    # plotEAGLE()
    # plotageFeH()
    # plot_scales_MgFeFeHvsFeH()
    # plot_data_MgFeFeHvsFeH()
    return 0


def makePlots(ESFwn=0, SMMwn=0, Zindex=1):
    EAGLE_FeHHist, EAGLEEdges = getEAGLE_hist_edges()
    G = Galaxy.loadFromBins(ESFweightingNum=ESFwn, NRG2SMMweightingNum=SMMwn)
    localD = Distributions('local', G.FeH(G.hist()), ISONumZIndex=Zindex)
    print("made localD")
    MWD = Distributions('MW', G.FeH(G.integratedHist()), perVolume=False, ISONumZIndex=Zindex)
    print("made MWD")
    EAGLED = Distributions('EAGLE', EAGLE_FeHHist, FeHEdges=EAGLEEdges, perVolume=False, ISONumZIndex=Zindex)
    print("made EAGLED")
    localD.plot(extra=f'ESFwn{ESFwn}SMMwn{SMMwn}Zindex{Zindex}')
    print("done local")
    MWD.plot(extra=f'ESFwn{ESFwn}SMMwn{SMMwn}Zindex{Zindex}')
    print("done MW")
    EAGLED.plot(extra=f'ESFwn{ESFwn}SMMwn{SMMwn}Zindex{Zindex}')
    print("done EAGLE")
    localD.plotWith(MWD, extra=f'ESFwn{ESFwn}SMMwn{SMMwn}Zindex{Zindex}')
    print("done localMW")
    MWD.plotWith(EAGLED, extra=f'ESFwn{ESFwn}SMMwn{SMMwn}Zindex{Zindex}', plotLim=(-2,1))
    print("done EAGLEMW")
    plotOverR(G, extra=f'ESFwn{ESFwn}SMMwn{SMMwn}Zindex{Zindex}')
    plotMedianOverR(G, extra=f'ESFwn{ESFwn}SMMwn{SMMwn}Zindex{Zindex}')
    plotFit(G, extra=f'ESFwn{ESFwn}SMMwn{SMMwn}')

def getEAGLE_hist_edges():
    EAGLE_data = np.loadtxt(os.path.join(mySetup.dataDir,
                            'input_data', 'EAGLE_MW_L0025N0376_REFERENCE_ApogeeRun_30kpc_working.dat'))
    # List of star particles
    EAGLE_mass = EAGLE_data[:,9]
    EAGLE_FeH = EAGLE_data[:,14]
    EAGLEEdges = mySetup.arr((-3.000, 1.000, 0.1))
    EAGLEWidths = EAGLEEdges[1:] - EAGLEEdges[:-1]
    EAGLE_FeHHist = np.histogram(EAGLE_FeH, bins=EAGLEEdges, weights=EAGLE_mass/EAGLEWidths[0])[0]
    
    return (EAGLE_FeHHist, EAGLEEdges)
    

def plotData(G, extra=''):
    fig, axs = plt.subplots(ncols=3, figsize=[18, 4])
    titles = ['N', 'mean R', 'mean mod z']
    for i, X in enumerate([G.data[0,:,:], G.data[1,:,:], G.data[2,:,:]]):
        image = axs[i].imshow(X.T, origin='lower', aspect='auto',
                              extent=(G.FeHEdges[0], G.FeHEdges[-1], G.aFeEdges[0], G.aFeEdges[-1]),
                              cmap=cmap1, norm=mpl.colors.LogNorm())
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(r'$\mathrm{[Fe/H]}$')
        axs[i].set_ylabel(r'$\mathrm{[\alpha/Fe]}$')
        fig.colorbar(image, ax=axs[i])
    fig.set_tight_layout(True)
    path = plotDir+str(extra)+'data.pdf'
    fig.savefig(path, dpi=300)
    

def plotFit(G, extra=None):
    binLim = 50
    fig, axs = plt.subplots(ncols=3, figsize=[18, 4])
    titles = ['amp', r'$a_R$', r'$a_z$']
    for i, X in enumerate([np.where(G.data[0,:,:]>=binLim, G.amp, 0),
                           np.where(G.data[0,:,:]>=binLim, G.aR, 0), 
                           np.where(G.data[0,:,:]>=binLim, G.az, 0)]):
        image = axs[i].imshow(X.T, origin='lower', aspect='auto',
                              extent=(G.FeHEdges[0], G.FeHEdges[-1], G.aFeEdges[0], G.aFeEdges[-1]),
                              cmap=cmap1, norm=mpl.colors.LogNorm())
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(r'$\mathrm{[Fe/H]}$')
        axs[i].set_ylabel(r'$\mathrm{[\alpha/Fe]}$')
        fig.colorbar(image, ax=axs[i])
    fig.set_tight_layout(True)
    path = plotDir+str(extra)+'fit.pdf'
    fig.savefig(path, dpi=300)

def plotAgeWeightingDiffs():
    binLim = 50
    
    G00 = Galaxy.loadFromBins(ESFweightingNum=0, NRG2SMMweightingNum=0)
    for k in range(1,3):
        G = Galaxy.loadFromBins(ESFweightingNum=k, NRG2SMMweightingNum=k)
        fig, axs = plt.subplots(ncols=3, figsize=[18, 4])
        titles = ['amp', r'$a_R$', r'$a_z$']
        for i, X in enumerate([np.where(G00.data[0,:,:]>=binLim, G.amp-G00.amp, 0),
                               np.where(G00.data[0,:,:]>=binLim, G.aR-G00.aR, 0), 
                               np.where(G00.data[0,:,:]>=binLim, G.az-G00.az, 0)]):
            image = axs[i].imshow(X.T, origin='lower', aspect='auto',
                                  extent=(G.FeHEdges[0], G.FeHEdges[-1], G.aFeEdges[0], G.aFeEdges[-1]),
                                  cmap=cmap1, norm=mpl.colors.Normalize())
            axs[i].set_title(titles[i])
            axs[i].set_xlabel(r'$\mathrm{[Fe/H]}$')
            axs[i].set_ylabel(r'$\mathrm{[\alpha/Fe]}$')
            fig.colorbar(image, ax=axs[i])
        fig.set_tight_layout(True)
        path = plotDir+str(k)+'DiffFit.pdf'
        fig.savefig(path, dpi=300)
        

def plotOverR(G, extra=''):
    R = np.linspace(0,20,101)
    R = (R[:-1]+R[1:])/2
    
    FeHPlotPoints = np.linspace(G.FeHEdges[0], G.FeHEdges[-1], 100)
    fH2OPlotPoints = np.linspace(fH2OLow+0.0001, fH2OHigh-0.0001, 100)
    FeHR = np.zeros((len(R), len(FeHPlotPoints)))
    fH2OR = np.zeros((len(R), len(fH2OPlotPoints)))
    
    for i, r in enumerate(R):
        D = Distributions(f'r={r}', G.FeH(G.zintegratedHist(r)), normalised=True)
        FeHR[i,:] = D.FeHDist(FeHPlotPoints)
        fH2OR[i,:] = D.fH2ODist(fH2OPlotPoints)
        
    fig, ax = plt.subplots()
    image = ax.imshow(FeHR.T, origin='lower',
              extent=(R[0], R[-1], FeHPlotPoints[0], FeHPlotPoints[-1]),
              aspect='auto', 
              cmap=cmap1, norm=mpl.colors.Normalize())
    fig.colorbar(image)
    ax.set_xlabel('R/kpc')
    ax.set_ylabel(r'FeH distribution')
    path = os.path.join(plotDir,str(extra)+'FeHR.pdf')
    fig.savefig(path, dpi=300)
    
    fig, ax = plt.subplots()
    image = ax.imshow(fH2OR.T, origin='lower',
              extent=(R[0], R[-1], fH2OPlotPoints[0], fH2OPlotPoints[-1]),
              aspect='auto', 
              cmap=cmap1, norm=mpl.colors.Normalize())
    fig.colorbar(image)
    ax.set_xlabel('R/kpc')
    ax.set_ylabel(r'fH2O distribution')
    path = os.path.join(plotDir,str(extra)+'fH2OR.pdf')
    fig.savefig(path, dpi=300)
    
    

def plotMedianOverR(G, extra=''):
    R = np.linspace(0,20,11)
    R = (R[:-1]+R[1:])/2
    FeH = np.zeros(len(R))
    fH2O = np.zeros(len(R))
    
    for i, r in enumerate(R):
        D = Distributions(f'r={r}', G.FeH(G.zintegratedHist(r)), normalised=True)
        FeH[i] = medianFeH(D)
        fH2O[i] = medianfH2O(D)
    
    fig, ax = plt.subplots()
    ax.plot(R, FeH)
    ax.set_xlabel('R/kpc')
    ax.set_ylabel(r'FeH median')
    path = os.path.join(plotDir,str(extra)+'FeHMedianR.pdf')
    fig.savefig(path, dpi=300)
    
    fig, ax = plt.subplots()
    ax.plot(R, fH2O)
    ax.set_xlabel('R/kpc')
    ax.set_ylabel(r'fH2O median')
    path = os.path.join(plotDir,str(extra)+'fH2OMedianR.pdf')
    fig.savefig(path, dpi=300)
    
def medianFeH(D):
    assert D.isNormalised
    def func(m):
        return scipy.integrate.quad(D.FeHDist, D.FeHEdges[0], m)[0]-0.5
    sol = scipy.optimize.root_scalar(func, bracket=(D.FeHEdges[0], D.FeHEdges[-1]))
    return sol.root

def medianfH2O(D):
    assert D.isNormalised
    if (D.counts[0]<0.5)and(D.counts[0]+D.counts[1]>0.5):
        def func(m):
            return D.counts[0]+scipy.integrate.quad(D.fH2ODist, fH2OLow, m)[0]-0.5
        sol = scipy.optimize.root_scalar(func, bracket=(fH2OLow, fH2OHigh))
        return sol.root
    else:
        # median not in middle range
        return np.nan #for now

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
    def loadFromBins(cls, FeHEdges=mySetup.FeHEdges, aFeEdges=mySetup.aFeEdges, ESFweightingNum=0, NRG2SMMweightingNum=0):
        """
        Note, can have different weighting in NRG2SMM and ESF for fit
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
        
        isogrid = myIsochrones.loadGrid()
        
        for i in range(shape[0]):
            isochrones = isogrid[(FeHEdges[i]<=isogrid['MH'])&(isogrid['MH']<FeHEdges[i+1])]
            NRG2SMM = myIsochrones.NRG2SMM(isochrones, NRG2SMMweightingNum)
            
            for j in range(shape[1]):
                binDir  = os.path.join(mySetup.dataDir, 'bins', f'FeH_{FeHEdges[i]:.3f}_{FeHEdges[i+1]:.3f}_aFe_{aFeEdges[j]:.3f}_{aFeEdges[j+1]:.3f}')
                
                
                with open(os.path.join(binDir, 'data.dat'), 'rb') as f0:
                    data[:,i,j] = np.array(pickle.load(f0))
                
                with open(os.path.join(binDir, f'w{ESFweightingNum}fit_results.dat'), 'rb') as f1:
                    logA, aR[i,j], az[i,j] = pickle.load(f1)
                    
                with open(os.path.join(binDir, f'w{ESFweightingNum}fit_sigmas.dat'), 'rb') as f1:
                    sig_logNuSun[i,j], sig_aR[i,j], sig_az[i,j] = pickle.load(f1)
                    
                    
                amp[i,j] = NRG2SMM*np.exp(logA)/vols[i,j]
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
        arg = np.where((self.az>0), -self.aR*(R-mySetup.R_Sun), 0)
        hist = np.where((self.az>0), 2*self.amp*np.exp(arg)/(self.az), 0)
        # volume technically infinite for negative a, this only occurs when negligible stars in bin
        # werid split here to avoid warnings
        
        if zlim!=None:
            hist *= (1 - np.exp(-self.az*zlim))
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.zintegratedHist())
        
    def FeH(self, hist):
        """
        From hist in FeH and aFe, integrates over aFe to get FeH alone
        """
        return ((hist*self.vols).sum(axis=1))/self.FeHWidths
        
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
    
    def __init__(self, name, FeHHist, FeHEdges=mySetup.FeHEdges, perVolume=True, normalised=False, ISONumZIndex=1):
        self.FeHEdges = FeHEdges
        self.FeHWidths = self.FeHEdges[1:] - self.FeHEdges[:-1]
        self.FeHMidpoints = (self.FeHEdges[1:] + self.FeHEdges[:-1])/2
        self.name = name
        self.perVolume = perVolume
        self.isNormalised = normalised
        if not normalised:
            self.FeHHist = FeHHist
        else:
            self.FeHHist = FeHHist/np.sum(self.FeHWidths*FeHHist)
        self.ISONumZIndex = ISONumZIndex
        
        y = np.append(0, np.cumsum(self.FeHWidths*self.FeHHist))
        dist = scipy.interpolate.CubicSpline(self.FeHEdges, y, bc_type='clamped', extrapolate=True).derivative()
        def FeHDistFunc(FeH):
            # sets dist to zero outside range
            return np.where((self.FeHEdges[0]<=FeH)&(FeH<self.FeHEdges[-1]), dist(FeH), 0)
        self.FeHDist = FeHDistFunc
        
        def ISOsPerFeH(FeH):
            return self.alpha*(10**(FeH*ISONumZIndex))*self.FeHDist(FeH)
            # return self.alpha*(((10**FeH)/(1+2.78*0.0207*(10**FeH)))**ISONumZIndex)*self.FeHDist(FeH)
        print(f"starting with {self.name}")
        time.sleep(2)
        lowerCount = scipy.integrate.quad(ISOsPerFeH, FeHHigh, self.FeHEdges[-1])[0] 
        middleCount = scipy.integrate.quad(ISOsPerFeH, FeHLow, FeHHigh)[0]
        upperCount = scipy.integrate.quad(ISOsPerFeH, self.FeHEdges[0], FeHLow)[0]
        normFactor = (lowerCount+middleCount+upperCount) if normalised else 1
        #Assumes bins entirely capture all stars
        
        time.sleep(2)
        print(f"ending with {self.name}, norm={self.isNormalised}")
        # quad has problem with EAGLE and some rs
        def fH2ODistFunc(fH2O):
            return ISOsPerFeH(compInv(fH2O))/(normFactor*np.abs(compDeriv(compInv(fH2O))))
        
        self.fH2ODist = fH2ODistFunc
        self.counts = (lowerCount/normFactor, middleCount/normFactor, upperCount/normFactor)
        
    def butNormalised(self):
        """returns a seperate Distributions with normalised=True"""
        # could this be cls() instead? No, that's for class methods, where cls is the first argument
        return self.__class__(self.name, self.FeHHist, self.FeHEdges, self.perVolume, normalised=True, ISONumZIndex=self.ISONumZIndex)
        
    def plot(self, extra='', plotLim=(None,None), saveDir=plotDir):
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
         
        
        
        FeHylab = ''
        fH2Oylab = ''
        fH2Ointylab = '' 
        
        
        FeHPlotPoints = np.linspace(self.FeHEdges[0], self.FeHEdges[-1], 10*len(self.FeHWidths))
        fH2OPlotPoints = np.linspace(fH2OLow+0.0001, fH2OHigh-0.0001)
        
        fig, ax = plt.subplots()
        ax.bar(self.FeHMidpoints, self.FeHHist, width = self.FeHWidths, alpha=0.5)
        ax.plot(FeHPlotPoints, self.FeHDist(FeHPlotPoints), color='C1')
        ax.vlines(-0.4, 0, self.FeHDist(0), color='C2', alpha=0.5)
        ax.vlines( 0.4, 0, self.FeHDist(0), color='C2', alpha=0.5)
        ax.set_xlabel(r'[Fe/H]')
        ax.set_xlim(plotLim[0], plotLim[1])
        ax.set_ylabel(FeHylab)
        ax.set_title(self.name)
        path = os.path.join(saveDir, str(extra) + self.name + '_FeH' + '.pdf')
        fig.savefig(path, dpi=300)    
        
        fig, ax = plt.subplots()
        ax.plot(fH2OPlotPoints, self.fH2ODist(fH2OPlotPoints))
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$f_\mathrm{H_2O}$')
        ax.set_ylabel(fH2Oylab)
        ax.set_title(self.name)
        path = os.path.join(saveDir, str(extra) + self.name + '_fH2O' + '.pdf')
        fig.savefig(path, dpi=300)    
        
        fig, ax = plt.subplots()
        ax.bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  self.counts)
        ax.set_ylabel(fH2Ointylab)
        ax.set_title(self.name)
        path = os.path.join(saveDir, str(extra) + self.name + '_bar' + '.pdf')
        fig.savefig(path, dpi=300)
          
        
        

    def plotWith(self, dists2, extra='', plotLim=(None,None), saveDir=plotDir):
        
        dists1 = self.butNormalised()
        dists2 = dists2.butNormalised()
        
        os.makedirs(saveDir, exist_ok=True)
        
        FeHylab = ''
        fH2Oylab = ''
        fH2Ointylab = '' 
        
        FeHPlotPoints = np.linspace(min(dists1.FeHEdges[0], dists2.FeHEdges[0]),
                                    max(dists1.FeHEdges[-1], dists2.FeHEdges[-1]),
                                    10*max(len(dists1.FeHWidths), len(dists2.FeHWidths)))
        fH2OPlotPoints = np.linspace(fH2OLow+0.0001, fH2OHigh-0.0001)
        
        names = f'{self.name}+{dists2.name}'
        
        fig, ax = plt.subplots()
        ax.plot(FeHPlotPoints, dists1.FeHDist(FeHPlotPoints), label=self.name)
        ax.plot(FeHPlotPoints, dists2.FeHDist(FeHPlotPoints), label=dists2.name)
        ax.vlines(-0.4, 0, dists1.FeHDist(0), color='C2', alpha=0.5)
        ax.vlines( 0.4, 0, dists1.FeHDist(0), color='C2', alpha=0.5)
        ax.legend()
        ax.set_xlabel(r'[Fe/H]')
        ax.set_xlim(plotLim[0], plotLim[1])
        ax.set_ylabel(FeHylab)
        ax.set_title(names)
        path = os.path.join(saveDir, str(extra) + names + '_FeH' + '.pdf')
        fig.savefig(path, dpi=300)
        
        fig, ax = plt.subplots()
        ax.plot(fH2OPlotPoints, dists1.fH2ODist(fH2OPlotPoints), label=self.name)
        ax.plot(fH2OPlotPoints, dists2.fH2ODist(fH2OPlotPoints), label=dists2.name)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.set_xlabel(r'$f_\mathrm{H_2O}$')
        ax.set_ylabel(fH2Oylab)
        ax.set_title(names)
        path = os.path.join(saveDir, str(extra) + names + '_fH2O' + '.pdf')
        fig.savefig(path, dpi=300)
        
        fig, ax = plt.subplots()
        ax.bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  dists1.counts, alpha=0.5, label=dists1.name)
        ax.bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  dists2.counts, alpha=0.5, label=dists2.name)
        ax.set_ylabel(fH2Ointylab)
        ax.legend()
        ax.set_title(names)
        path = os.path.join(saveDir, str(extra) + names + '_bar' + '.pdf')
        fig.savefig(path, dpi=300)
        
                    
        
    

# Defining comp:
FeH_p = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
compPoly = np.polynomial.polynomial.Polynomial.fit(FeH_p, fH2O_p, 3)
FeHLow = FeH_p[0]
FeHHigh = FeH_p[-1]
fH2OLow = compPoly(FeH_p[-1])
fH2OHigh = compPoly(FeH_p[0])
def comp(FeH):
    return np.where(FeHLow<=FeH, np.where(FeH<FeHHigh,
                                          compPoly(FeH),
                                          fH2OLow), fH2OHigh)
    
def compInv(fH2O):
    """inv may not work with array inputs"""
    if np.ndim(fH2O)==0:
        val = fH2O
        if fH2OLow<=val<=fH2OHigh:
            allroots = (compPoly-val).roots()
            myroot = allroots[(FeH_p[0]<=allroots)&(allroots<=FeH_p[-1])]
            assert len(myroot)==1 # checks not multiple roots
            assert np.isreal(myroot[0])
            return np.real(myroot[0])
        else:
            return np.nan
    else:
        returnArray = np.zeros_like(fH2O)
        for i, val in enumerate(fH2O):
            if fH2OLow<=val<fH2OHigh:
                allroots = (compPoly-val).roots()
                myroot = allroots[(FeH_p[0]<=allroots)&(allroots<FeH_p[-1])]
                assert len(myroot)==1 # checks not multiple roots
                assert np.isreal(myroot[0])
                returnArray[i] = np.real(myroot[0])
            else:
                returnArray[i] =  np.nan
        return returnArray
        
def compDeriv(FeH):
    return np.where((FeHLow<=FeH)&(FeH<FeHHigh), compPoly.deriv()(FeH), 0)

for x in [-0.4+0.0001, -0.2, 0, 0.2, 0.4-0.0001]:
    assert np.isclose(compInv(comp(x)), x) #checks inverse works
    
    
    
    
if __name__=='__main__':
    main()
    
    
    
    
    
    