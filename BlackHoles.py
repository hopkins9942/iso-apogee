import time
from math import isclose

import numpy as np
import os
import pickle
import git

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

from scipy.integrate import quad 

import mySetup
import myIsochrones


# For black holes letter


cmap0 = mpl.colormaps['Blues']
cmap01 = mpl.colormaps['Purples']
cmap1 = mpl.colormaps['Greys']#mpl.colormaps['Blues']
cmap2 = mpl.colormaps['hsv']
# colourPalette = mpl.colormaps['tab10'](np.linspace(0.05, 0.95, 10))
colourPalette = [ 'goldenrod','darkslateblue', 'teal', 'red']

plt.rcParams.update({
    "text.usetex": True})

plotDir = r'D:\Moved folders\OneDrive\OneDrive\Documents\Code\APOGEE\plots'
# plotDir = f'/Users/hopkinsm/APOGEE/plots/BlackHoles/'
#f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/'
#plotDir = '/home/hopkinsl/Documents/APOGEE/plots'
os.makedirs(plotDir, exist_ok=True)

rBH = 1.48e-3 # calculated by hand, integrating Kroupa from 0.08 to inf vs 25 to inf
# rBH = 6.5e-3 integrating upwars of 8Msun, which is all remnants



def main():
    G = Galaxy.loadFromBins()
    
    # plotFit(G, G.mask())
    totalBHs(G)
    # ageOverR(G)
    
    # plotmidplane(G)
    # plotOverR(G)
    # print(F1(G))
    # KSdiff(G, 20)
    
def totalBHs(G):
    rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    # starcounts = G.FeH(G.integratedHist(Rlim1=0), G.mask())*G.FeHWidths 
    starcounts = G.FeH(G.hist(), G.mask())*G.FeHWidths # aroudnSun - per kpc3?
    
    BHcount = (rBHarray*starcounts).sum()
    # BHcount = (rBH*starcounts).sum() for no metallicity cutoff
    smCount = starcounts.sum()
    print(BHcount, smCount, BHcount/smCount)

def ageOverR(G):
    """I measure the SM, LS and RG surface density and mean age at each R,
    then for ratios between types of star compare full measured and predicted 
    from mean age alone, to check fact that SM/LS barely varies over R.
    result: mean age clearly varies over R, from 6.4 to 4.8 Gyr. Using
    this mean age to estimate the ratio of SM to RG overpredicts true value
    (because curvature of NRG2N__ is negative - middle value of age
     overpredicts mean of spread - therefore fine) and since overprediction is
    slightly less for ls, prediction for ls2sm overestimates too.
    HOWEVER, using mean tau, which looks right, still gives very small variation
    in LS2SM, meaning very small variation in actual measured is fine. 
    Conclusion: LS2SM jsut doesn't vary much over disk"""
    R = np.linspace(4,12,51)
    R = (R[:-1]+R[1:])/2
    tau = np.linspace(0,14,51)
    tau = (tau[:-1]+tau[1:])/2
    
    isogrid = myIsochrones.loadGrid()
    isochrones = isogrid[(-0.5<=isogrid['MH'])&(isogrid['MH']<0)]
    
    SMstars = np.zeros(len(R))
    LSstars = np.zeros(len(R))
    RGstars = np.zeros(len(R))
    meanTau0 = np.zeros(len(R))
    rg2sm = np.zeros(len(R))
    rg2ls = np.zeros(len(R))
    ageDist = np.zeros((len(R), len(tau)))
    
    for i, r in enumerate(R):
        SMstarFeHcounts = (G.FeH(G.zintegratedHist(r), G.mask())*G.FeHWidths)
        LSstarFeHcounts = (G.FeH(G.zintegratedHist(r)*G.NRG2NLS/G.NRG2NSM, G.mask())*G.FeHWidths)
        RGstarFeHcounts = (G.FeH(G.zintegratedHist(r)/G.NRG2NSM, G.mask())*G.FeHWidths)
        tau0FeHsums = (G.FeH(G.zintegratedHist(r)*G.tau0, G.mask())*G.FeHWidths)
        SMstars[i] = SMstarFeHcounts.sum()
        LSstars[i] = LSstarFeHcounts.sum()
        RGstars[i] = RGstarFeHcounts.sum()
        meanTau0[i] = tau0FeHsums.sum()/SMstars[i]
        for j in range(len(tau)):
            ageDist[i,j] = (G.FeH(G.zintegratedHist(r)*np.sqrt(G.omega/(2*np.pi))
                                  *np.exp(-G.omega*(tau[j]-G.tau0)**2/2),
                                  G.mask())*G.FeHWidths).sum()
        ageDist[i,:]/=ageDist[i,:].sum()
        
        rg2sm[i] = myIsochrones.NRG2NSM(isochrones, meanTau0[i], 1)
        rg2ls[i] = myIsochrones.NRG2NLS(isochrones, meanTau0[i], 1)
    
    overage_rg2sm = np.zeros(len(tau))
    overage_rg2ls = np.zeros(len(tau))
    for j in range(len(tau)):
        overage_rg2sm[j] = myIsochrones.NRG2NSM(isochrones, tau[j], 1)
        overage_rg2ls[j] = myIsochrones.NRG2NLS(isochrones, tau[j], 1)
        
    fig, ax = plt.subplots()
    ax.plot(tau, overage_rg2sm)
    ax.plot(tau, overage_rg2ls)
    ax.set_xlabel(r'age / Gyr')
    ax.set_ylabel(r'rg2_')
    # path = os.path.join(plotDir, '.pdf')
    # fig.tight_layout()
    # fig.savefig(path, dpi=300)
    
    fig, ax = plt.subplots()
    ax.plot(R, meanTau0)
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'mean tau0 / Gyr')
    path = os.path.join(plotDir, 'tau0overR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    
    fig,ax = plt.subplots()
    ax.imshow(ageDist.T, origin='lower',
              extent=(R[0], R[-1], tau[0], tau[-1]),
              aspect='auto')
    
    fig, ax = plt.subplots()
    ax.plot(R, SMstars/RGstars, label='full measured')
    ax.plot(R, rg2sm, label='estimated from mean age')
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'rg2sm')
    fig.legend()
    path = os.path.join(plotDir, 'rg2smOverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    
    fig, ax = plt.subplots()
    ax.plot(R, LSstars/RGstars, label='full measured')
    ax.plot(R, rg2ls, label='estimated from mean age')
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'rg2ls')
    fig.legend()
    path = os.path.join(plotDir, 'rg2lsOverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    
    fig, ax = plt.subplots()
    ax.plot(R, SMstars/LSstars, label='full measured')
    ax.plot(R, rg2sm/rg2ls, label='estimated from mean age')
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'ls2sm')
    fig.legend()
    path = os.path.join(plotDir, 'ls2smOverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    

def F1(G):
    """for now ignored BDs (0.05 fraction error) and midplane and assuming
    average mass of BH is 10"""
    
    R = np.linspace(0,12,76)
    R = (R[:-1]+R[1:])/2
    
    rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    
    SMstars = np.zeros(len(R))
    LSstars = np.zeros(len(R))
    BHs = np.zeros(len(R))
    
    for i, r in enumerate(R):
        SMstarFeHcounts = (G.FeH(G.hist(r), G.mask())*G.FeHWidths)
        LSstarFeHcounts = (G.FeH(G.hist(r)*G.NRG2NLS/G.NRG2NSM, G.mask())*G.FeHWidths)
        SMstars[i] = SMstarFeHcounts.sum() 
        LSstars[i] = LSstarFeHcounts.sum() 
        BHs[i] = (SMstarFeHcounts*rBHarray).sum()
        
    x = np.linspace(0,1,76)
    x = (x[:-1]+x[1:])/2
    
    F1 = (10/myIsochrones.meanMini)*(np.sum(BHs*x*(1-x))/np.sum(LSstars*x*(1-x)))
    
    return F1
    
def plotmidplane(G):
    
    R = np.linspace(0,12,76)
    R = (R[:-1]+R[1:])/2
    
    rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    
    SMstars = np.zeros(len(R))
    LSstars = np.zeros(len(R))
    BHs = np.zeros(len(R))
    
    for i, r in enumerate(R):
        SMstarFeHcounts = (G.FeH(G.hist(r), G.mask())*G.FeHWidths)
        LSstarFeHcounts = (G.FeH(G.hist(r)*G.NRG2NLS/G.NRG2NSM, G.mask())*G.FeHWidths)
        SMstars[i] = SMstarFeHcounts.sum() 
        LSstars[i] = LSstarFeHcounts.sum() 
        BHs[i] = (SMstarFeHcounts*rBHarray).sum()
        
        
    fig, ax = plt.subplots()
    ax.plot(R, LSstars, label='Living stars')
    ax.plot(R, SMstars, label='SM stars')
    ax.plot(R, BHs*1e3, label=r'BHs$\cdot 10^3$')
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'midplane density /$\mathrm{number}\;\mathrm{kpc}^{-3}$')
    # ax.set_yscale('log')
    fig.legend()
    path = os.path.join(plotDir, 'midoverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    
    fig, ax = plt.subplots()
    ax.plot(R[R>3.95], BHs[R>3.95]*1e3/LSstars[R>3.95], 'C0', label='BHs/ 1000 living stars')
    ax.plot(R[R<4], BHs[R<4]*1e3/LSstars[R<4], '--C0')
    # ax.plot(R, BHs*1e3/LSstars, label=r'BHs$\cdot 10^3$ / living stars')
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'volume density ratio')
    ax.set_ylim([0, 1.5])
    fig.legend()
    path = os.path.join(plotDir, 'midratiosoverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    # print(SMstars/LSstars)
    
def plotOverR(G):
    R = np.linspace(0,12,51)
    R = (R[:-1]+R[1:])/2
    
    rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    
    SMstars = np.zeros(len(R))
    LSstars = np.zeros(len(R))
    BHs = np.zeros(len(R))
    
    for i, r in enumerate(R):
        SMstarFeHcounts = (G.FeH(G.zintegratedHist(r), G.mask())*G.FeHWidths)
        LSstarFeHcounts = (G.FeH(G.zintegratedHist(r)*G.NRG2NLS/G.NRG2NSM, G.mask())*G.FeHWidths)
        SMstars[i] = SMstarFeHcounts.sum() 
        LSstars[i] = LSstarFeHcounts.sum() 
        BHs[i] = (SMstarFeHcounts*rBHarray).sum()
        
        
    fig, ax = plt.subplots()
    ax.plot(R[R>3.95], LSstars[R>3.95], 'C0', label='Living stars')
    ax.plot(R[R<4], LSstars[R<4], '--C0')
    ax.plot(R[R>3.95], SMstars[R>3.95],'C1', label='SM stars')
    ax.plot(R[R<4], SMstars[R<4], '--C1')
    ax.plot(R[R>3.95], BHs[R>3.95]*1e3, 'C2', label=r'BHs$\cdot 10^3$')
    ax.plot(R[R<4], BHs[R<4]*1e3, '--C2')
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'Surface density /$\mathrm{number}\;\mathrm{kpc}^{-2}$')
    # ax.set_yscale('log')
    fig.legend()
    path = os.path.join(plotDir, 'overR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    
    fig, ax = plt.subplots()
    ax.plot(R[R>3.95], BHs[R>3.95]*1e3/LSstars[R>3.95], 'C0', label='BHs/ 1000 living stars')
    ax.plot(R[R<4], BHs[R<4]*1e3/LSstars[R<4], '--C0')
    # ax.plot(R, BHs*1e3/LSstars, label=r'BHs$\cdot 10^3$ / living stars')
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'Surface density ratio')
    ax.set_ylim([0, 1.5])
    fig.legend()
    path = os.path.join(plotDir, 'ratiosoverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    # print(SMstars/LSstars)
    
    
def KSdiff(G,N):
    """ finds how many draws from BH ZINTEGRATED dist needed to get 5% KS test to stellar dist
    """
    
    
    R = np.linspace(0,8,51)
    R = (R[:-1]+R[1:])/2
    
    rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    
    SMstars = np.zeros(len(R))
    LSstars = np.zeros(len(R))
    BHs = np.zeros(len(R))
    
    for i, r in enumerate(R):
        SMstarFeHcounts = (G.FeH(G.hist(r), G.mask())*G.FeHWidths)
        LSstarFeHcounts = (G.FeH(G.hist(r)*G.NRG2NLS/G.NRG2NSM, G.mask())*G.FeHWidths)
        SMstars[i] = SMstarFeHcounts.sum() 
        LSstars[i] = LSstarFeHcounts.sum() 
        BHs[i] = (SMstarFeHcounts*rBHarray).sum()
    
    BHcs = np.cumsum(BHs)
    BHcs/=BHcs[-1]
    LScs = np.cumsum(LSstars)
    LScs/=LScs[-1]
    
    
    LSdist = lambda r: LScs[np.digitize(r,R)]
    
    p = np.array([])
    
    for i in range(100):
        draws = np.random.uniform(size=N)
        BHRs = R[np.digitize(draws,BHcs)]-0.0001
        p = np.append(p, scipy.stats.kstest(BHRs, LSdist).pvalue)
    
    return (p.mean(), p.std())
    

def plotFit(G, mask, extra=''):
    savename = ['logNuSun',  'aR', 'az', 'tau0', 'omegahalf', 'NRG2NSM', 'NRG2NLS', 'SMperLS']
    titles = [r'$\mathrm{logA}$', r'$a_R$', r'$a_z$', r'$\tau_0$', r'$\omega^{-\frac{1}{2}}$', r'$n_\mathrm{sm}/n_\mathrm{giants}$', r'$n_\mathrm{ls}/n_\mathrm{giants}$', r'$n_\mathrm{sm}/n_\mathrm{living}$']
    unit = [r'', r'$\mathrm{kpc}^{-1}$', r'$\mathrm{kpc}^{-1}$', r'$\mathrm{Gyr}$', r'$\mathrm{Gyr}$', r'', '', '']
    ageMask = mask*(np.arange(G.shape[0])>=15).reshape(-1,1)
    for i, X in enumerate([np.where(mask, G.logNuSun, np.nan),
                           np.where(mask, G.aR,    np.nan), 
                           np.where(mask, G.az,    np.nan),
                           np.where(ageMask, G.tau0,  np.nan), 
                           np.where(ageMask, G.omega**-0.5, np.nan), 
                           np.where(mask, G.NRG2NSM, np.nan), 
                           np.where(mask, G.NRG2NLS, np.nan), 
                           np.where(mask, G.NRG2NSM/G.NRG2NLS, np.nan)]):
        fig, ax = plt.subplots()
        image = ax.imshow(X.T, origin='lower', aspect='auto',
                              extent=(G.FeHEdges[0], G.FeHEdges[-1], G.aFeEdges[0], G.aFeEdges[-1]),
                              cmap=cmap0)#, norm=mpl.colors.LogNorm())
        ax.set_title(titles[i])
        ax.set_xlabel(r'$\mathrm{[Fe/H]}$')
        ax.set_ylabel(r'$\mathrm{[\alpha/Fe]}$')
        ax.set_facecolor("gainsboro")#("lavenderblush")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(unit[i])
        # cbar.set_label('' if i==0 else r'$\mathrm{kpc}^{-1}$')
        fig.set_tight_layout(True)
        path = plotDir+'/'+str(extra)+str(savename[i])+'fit.pdf'
        fig.savefig(path, dpi=300)




class Galaxy:
    def __init__(self, FeHEdges, aFeEdges, rhoSun, logNuSun, aR, az, tau0, omega, sig_logNuSun, sig_aR, sig_az, sig_tau0, sig_omega, NRG2NSM, NRG2NLS, data):
        """
        amp etc are arrays with [FeH index, aFe index]
        
        logNuSun is what comes out of the fit, the cartesian space number density of red giants at R=R0, z=0 in each bin
        rhoSun is this multiplied by NRG2SMM, divided by volume of bin, so equals the denisty in space and composition of SM mass distribution
        
        Difference to analysis.py - uses NRG2NSM to get number density of sine morte population
        """
        self.FeHEdges = FeHEdges
        self.aFeEdges = aFeEdges
        self.rhoSun = rhoSun
        self.logNuSun = logNuSun # logNuSun from fit, for testing
        self.aR = aR
        self.az = az
        self.tau0 = tau0
        self.omega = omega
        self.sig_logNuSun = sig_logNuSun # unsure if useful
        self.sig_aR = sig_aR
        self.sig_az = sig_az
        self.sig_tau0 = sig_tau0
        self.sig_omega = sig_omega
        self.NRG2NSM = NRG2NSM
        self.NRG2NLS = NRG2NLS
        self.data = data
        
        self.shape = self.rhoSun.shape
        self.FeHWidths = FeHEdges[1:] - FeHEdges[:-1]
        self.FeHMidpoints = (FeHEdges[1:] + FeHEdges[:-1])/2
        self.aFeWidths = aFeEdges[1:] - aFeEdges[:-1]
        self.aFeMidpoints = (aFeEdges[1:] + aFeEdges[:-1])/2
        self.vols = self.FeHWidths.reshape(-1,1) * self.aFeWidths
        assert self.vols.shape==self.rhoSun.shape
        
        
    def mask(self, N=20, s=1):# err part currently completely flawed by negative aR
        return (
                (self.data[0,:,:]         >=N)
                )
    
    @classmethod
    def loadFromBins(cls, FeHEdges=mySetup.FeHEdges, aFeEdges=mySetup.aFeEdges, fitLabel='lowFeHUniform+Rzlim+plotwithoutnan'):
        """
        
        """
        shape = (len(FeHEdges)-1, len(aFeEdges)-1)
        rhoSun = np.zeros(shape)#amp of rho
        logNuSun = np.zeros(shape)#log  numbr desnity of giants
        aR = np.zeros(shape)
        az = np.zeros(shape)
        tau0 = np.zeros(shape)
        omega = np.zeros(shape)
        sig_logNuSun = np.zeros(shape)
        sig_aR = np.zeros(shape)
        sig_az = np.zeros(shape)
        sig_tau0 = np.zeros(shape)
        sig_omega = np.zeros(shape)
        NRG2NSM = np.zeros(shape)
        NRG2NLS = np.zeros(shape)
        data = np.zeros((5, *shape))
        
        FeHWidths = FeHEdges[1:] - FeHEdges[:-1]
        aFeWidths = aFeEdges[1:] - aFeEdges[:-1]
        vols = FeHWidths.reshape(-1,1) * aFeWidths
        
        isogrid = myIsochrones.loadGrid()
        
        for i in range(shape[0]):
            isochrones = isogrid[(FeHEdges[i]<=isogrid['MH'])&(isogrid['MH']<FeHEdges[i+1])]

            
            for j in range(shape[1]):
            
                binDir  = os.path.join(mySetup.dataDir, 'bins', f'FeH_{FeHEdges[i]:.3f}_{FeHEdges[i+1]:.3f}_aFe_{aFeEdges[j]:.3f}_{aFeEdges[j+1]:.3f}')
                
                with open(os.path.join(binDir, 'data.dat'), 'rb') as f0:
                    data[:,i,j] = np.array(pickle.load(f0))
                    
                with open(os.path.join(binDir, fitLabel+'fit_results.dat'), 'rb') as f1:
                    logNuSun[i,j], aR[i,j], az[i,j], tau0[i,j], omega[i,j] = pickle.load(f1)
                        
                    
                if FeHEdges[0]>-0.55:
                    with open(os.path.join(binDir, fitLabel+'fit_sigmas.dat'), 'rb') as f2:
                        sig_logNuSun[i,j], sig_aR[i,j], sig_az[i,j], sig_tau0[i,j], sig_omega[i,j] = pickle.load(f2)
                
                else:
                    with open(os.path.join(binDir, fitLabel+'fit_sigmas.dat'), 'rb') as f2:
                        sig_logNuSun[i,j], sig_aR[i,j], sig_az[i,j], sig_tau0[i,j], sig_omega[i,j] = pickle.load(f2)
                        
                if data[0,i,j] !=0:
                    if FeHEdges[i]<-0.55: #low Fe
                        NRG2NSM[i,j] = myIsochrones.NRG2NSM(isochrones, 12, 1) #uses old age to get upper lim 
                        NRG2NLS[i,j] = myIsochrones.NRG2NLS(isochrones, 12, 1) #uses old age to get upper lim 
                    else:
                        NRG2NSM[i,j] = myIsochrones.NRG2NSM(isochrones, tau0[i,j], omega[i,j]) 
                        NRG2NLS[i,j] = myIsochrones.NRG2NLS(isochrones, tau0[i,j], omega[i,j]) 
                    
                    rhoSun[i,j] = NRG2NSM[i,j]*np.exp(logNuSun[i,j])/vols[i,j]
                else:
                    rhoSun[i,j] = 0
                    
        return cls(FeHEdges, aFeEdges, rhoSun, logNuSun, aR, az, tau0, omega, sig_logNuSun, sig_aR, sig_az, sig_tau0, sig_omega, NRG2NSM, NRG2NLS, data)
    
    
    def hist(self, R=mySetup.R_Sun, z=mySetup.z_Sun, normalised=False):
        hist = self.rhoSun*np.exp( - self.aR*(R-mySetup.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.hist(R,z)) #assumes bins cover whole distribution
    

    def integratedHist(self, Rlim1=4, Rlim2=12, zlim=5, normalised=False):
        """integrates R=0 to R and z=-z to z, with default of whole galaxy"""
        # hist = np.where((self.aR>0)&(self.az>0), 4*np.pi*self.rhoSun*np.exp(self.aR*mySetup.R_Sun)/(self.aR**2 * self.az), 0) removed now using R and z limits
        # volume technically infinite for negative a, this only occurs when negligible stars in bin
        # if Rlim!=None:
        #     hist *= (1 - (1+self.aR*Rlim)*np.exp(-self.aR*Rlim))
        # if zlim!=None:
        #     hist *= (1 - np.exp(-self.az*zlim))
        
        hist = ((4*np.pi*self.rhoSun*np.exp(self.aR*mySetup.R_Sun)/(self.aR**2 * self.az))
                * ((1+self.aR*Rlim1)*np.exp(-self.aR*Rlim1) - (1+self.aR*Rlim2)*np.exp(-self.aR*Rlim2))
                * (1 - np.exp(-self.az*zlim)))
            
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.integratedHist())
        

    def zintegratedHist(self, R=mySetup.R_Sun, zlim=5, normalised=False):
        """integrates z=-z to z at given R"""
        # arg = np.where((self.az>0), -self.aR*(R-mySetup.R_Sun), 0)
        # hist = np.where((self.az>0), 2*self.rhoSun*np.exp(arg)/(self.az), 0) removed now using R and z limits and allowing negative aRaz
        # volume technically infinite for negative a, this only occurs when negligible stars in bin
        # werid split here to avoid warnings
        
        # if zlim!=None:
        #     hist *= (1 - np.exp(-self.az*zlim))
        
        
        hist = (2*self.rhoSun*np.exp(-self.aR*(R-mySetup.R_Sun))/(self.az))*(1 - np.exp(-self.az*zlim))
        
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.zintegratedHist())
        

    def FeH(self, hist, mask):
        """
        From hist in FeH and aFe, integrates over aFe to get FeH alone
        
        useMask avoids bins with uncertain parameter values 
        """
        
        return ((np.where(mask, hist, 0)*self.vols).sum(axis=1))/self.FeHWidths
    
    
    
    
if __name__=='__main__':
    main()
    
    
    
    
    
    