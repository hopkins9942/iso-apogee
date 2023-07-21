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

plotDir = f'/Users/hopkinsm/APOGEE/plots/BlackHoles/'
#f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/'
#plotDir = '/home/hopkinsl/Documents/APOGEE/plots'
os.makedirs(plotDir, exist_ok=True)

rBH = 1.48e-3 # calculated by hand, integrating Kroupa from 0.08 to inf vs 25 to inf


def main():
    G = Galaxy.loadFromBins()
    # print(G.NRG2NSM[G.mask()])
    
    # plotFit(G, G.mask())
    
    # # Black holes made for stars of mass>25 MSun, metallicity less than 0, inspired by
    # # https://iopscience.iop.org/article/10.1086/375341/pdf
    # # assuming kroupa imf
    
    # # fraction of stars over 25 Msun
    
    # rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    # print(rBHarray)
    
    # starcounts = G.FeH(G.hist(), G.mask())*G.FeHWidths # aroudnSun - per kpc3?
    # fig,ax = plt.subplots()
    # ax.plot(G.FeHMidpoints, starcounts/G.FeHWidths)
    # bhcounts =starcounts*rBHarray
    # fig,ax = plt.subplots()
    # ax.plot(G.FeHMidpoints, bhcounts/G.FeHWidths)
    # # sum counts for density/number of bh
    
    # BH gradient
    plotOverR(G)
    
    
def plotOverR(G):
    R = np.linspace(4,12,51)
    R = (R[:-1]+R[1:])/2
    
    rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    
    stars = np.zeros(len(R))
    BHs = np.zeros(len(R))
    
    for i, r in enumerate(R):
        starFeHcounts = (G.FeH(G.zintegratedHist(r), G.mask())*G.FeHWidths) 
        stars[i] = starFeHcounts.sum() 
        BHs[i] = (starFeHcounts*rBHarray).sum()
        
    fig, ax = plt.subplots()
    ax.plot(R, stars, label='stars')
    ax.plot(R, BHs*1e3, label=r'BHs$\cdot 10^3$')
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'Surface density /$\mathrm{number}\;\mathrm{kpc}^{-3}$')
    # ax.set_yscale('log')
    fig.legend()
    path = os.path.join(plotDir, 'overR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    
    
    

def plotFit(G, mask, extra=''):
    savename = ['logNuSun',  'aR', 'az', 'tau0', 'omegahalf', 'NRG2NSM']
    titles = [r'$\mathrm{logA}$', r'$a_R$', r'$a_z$', r'$\tau_0$', r'$\omega^{-\frac{1}{2}}$', r'$n_\mathrm{sm}/n_\mathrm{giants}$']
    unit = [r'', r'$\mathrm{kpc}^{-1}$', r'$\mathrm{kpc}^{-1}$', r'$\mathrm{Gyr}$', r'$\mathrm{Gyr}$', r'']
    ageMask = mask*(np.arange(G.shape[0])>=15).reshape(-1,1)
    for i, X in enumerate([np.where(mask, G.logNuSun, np.nan),
                           np.where(mask, G.aR,    np.nan), 
                           np.where(mask, G.az,    np.nan),
                           np.where(ageMask, G.tau0,  np.nan), 
                           np.where(ageMask, G.omega**-0.5, np.nan), 
                           np.where(mask, G.NRG2NSM, np.nan)]):
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
    def __init__(self, FeHEdges, aFeEdges, rhoSun, logNuSun, aR, az, tau0, omega, sig_logNuSun, sig_aR, sig_az, sig_tau0, sig_omega, NRG2NSM, data):
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
                        # NRG2SMM[i,j] = myIsochrones.NRG2SMM(isochrones, 7, 0.0001) #uses uniform age 
                        NRG2NSM[i,j] = myIsochrones.NRG2NSM(isochrones, 12, 1) #uses old age to get upper lim 
                    else:
                        NRG2NSM[i,j] = myIsochrones.NRG2NSM(isochrones, tau0[i,j], omega[i,j]) 
                    
                    rhoSun[i,j] = NRG2NSM[i,j]*np.exp(logNuSun[i,j])/vols[i,j]
                else:
                    rhoSun[i,j] = 0
                    
        return cls(FeHEdges, aFeEdges, rhoSun, logNuSun, aR, az, tau0, omega, sig_logNuSun, sig_aR, sig_az, sig_tau0, sig_omega, NRG2NSM, data)
    
    
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
    
    
    
    
    
    