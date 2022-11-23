
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

POLYDEG = 3

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:7]
print(repo)
print(sha)

    
def main():
    # plotFeH()
    plotMgFeFeH()
    # plotageFeH()
    # plot_scales_MgFeFeHvsFeH()
    # plot_data_MgFeFeHvsFeH()
    
    
    
def plot_data_MgFeFeHvsFeH():
    
    saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/analysis_data/'
    os.makedirs(saveDir, exist_ok=True)
    
    FeHEdges = myUtils._FeH_edges # assumes same FeH bins for both fits. could do differently
    MgFeEdges = myUtils._MgFe_edges 
    
    G_MgFeFeH = Galaxy.loadFromBins(['FeH', 'MgFe'], [FeHEdges,MgFeEdges], noPyro=True)
    lenFeH, lenMgFe = G_MgFeFeH.shape
    
    G_FeH = Galaxy.loadFromBins(['FeH',], [FeHEdges,])
    
    FeH_axis = G_MgFeFeH.labels.index('FeH')
    assert FeH_axis==0
    
    Nlim = 5
    
    labels=['N','mean R', 'mean modz']
    for k in range(3):
        fig, ax = plt.subplots()
        for i in range(lenFeH):
            if G_FeH.data[0,i]>=Nlim:
                ax.scatter(G_FeH.midpoints[0][i], G_FeH.data[k,i], color='C0', alpha=0.5)
                for j in range(lenMgFe):
                    # print(G_FeH.midpoints[0][i], G_MgFeFeH.data[k,i,j])
                    if G_MgFeFeH.data[0,i,j]>=Nlim:
                        
                        ax.scatter(G_FeH.midpoints[0][i], G_MgFeFeH.data[k,i,j], color='C1', alpha=0.5)
                if k==0:
                    ax.scatter(G_FeH.midpoints[0][i], G_MgFeFeH.data[k,i,:].sum(), color='C2', alpha=0.5)
        if k != 1: ax.set_yscale('log')
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel(f'{labels[k]}')
        path = saveDir+f'{labels[k]}'
        fig.savefig(path, dpi=300)
    
    
    
def plot_scales_MgFeFeHvsFeH():
    
    saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/analysis_scales/'
    os.makedirs(saveDir, exist_ok=True)
    
    
    
    FeHEdges = myUtils._FeH_edges # assumes same FeH bins for both fits. could do differently
    MgFeEdges = myUtils._MgFe_edges 
    
    G_MgFeFeH = Galaxy.loadFromBins(['FeH', 'MgFe'], [FeHEdges,MgFeEdges], noPyro=True)
    lenFeH, lenMgFe = G_MgFeFeH.shape
    print(G_MgFeFeH.shape)
    
    G_FeH = Galaxy.loadFromBins(['FeH',], [FeHEdges,])
    
    FeH_axis = G_MgFeFeH.labels.index('FeH')
    assert FeH_axis==0
    
    print(1/G_MgFeFeH.aR[-5,:])
    print(1/G_FeH.aR[-5])
    print(G_FeH.midpoints[0])
    print(G_FeH.data[:,-5])
    
    Nlim=5
    
    
    # for i in range(lenFeH):
    #     if G_FeH.data[0][i]>=Nlim:
    #         fig, ax = plt.subplots()
    #         x = 1/G_FeH.aR[i]
    #         y = 1/G_FeH.az[i]
    #         ax.scatter(x, y, color='C0', alpha=0.5)
    #         ax.annotate(f'{G_FeH.data[0][i]:.0f}', (x, y))
    #         for j in range(lenMgFe):
    #             if G_MgFeFeH.data[0][i,j]>=Nlim:
    #                 x = 1/G_MgFeFeH.aR[i,j]
    #                 y = 1/G_MgFeFeH.az[i,j]
    #                 ax.scatter(x, y, color='C1', alpha=0.5)
    #                 ax.annotate(f'{G_MgFeFeH.data[0][i,j]:.0f}', (x, y))
    #         # ax.set_xscale('log')
    #         # ax.set_yscale('log')
    #         ax.set_xlabel('scale length /kpc')
    #         ax.set_ylabel('scale height /kpc')
    #         ax.set_title(f'FeH={G_FeH.edges[0][i]:.3f} to {G_FeH.edges[0][i+1]:.3f}')
            
    
    
    MgFeTotalWidth = G_MgFeFeH.edges[1][-1]-G_MgFeFeH.edges[1][0]
    
    
    fig, ax = plt.subplots()
    for i in range(lenFeH):
        if G_FeH.data[0][i]>=Nlim:
            ax.scatter(G_FeH.midpoints[0][i], G_FeH.hist()[i]/MgFeTotalWidth, color='C0', alpha=0.5)
            for j in range(lenMgFe):
                if G_MgFeFeH.data[0][i,j]>=Nlim:
                    ax.scatter(G_FeH.midpoints[0][i], G_MgFeFeH.hist()[i,j], color='C1', alpha=0.5)
            ax.scatter(G_FeH.midpoints[0][i], (G_MgFeFeH.widths[1]*G_MgFeFeH.hist()[i,:]).sum()/MgFeTotalWidth, color='C2', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('Mass density at Sun per [Fe/H] per [Mg/Fe] /Msol pc^-3 dex^-2')
    path = saveDir+'amp'
    fig.savefig(path, dpi=300)
            
    fig, ax = plt.subplots()
    for i in range(lenFeH):
        if G_FeH.data[0][i]>=Nlim:
            ax.scatter(G_FeH.midpoints[0][i], G_FeH.integratedHist()[i]/MgFeTotalWidth, color='C0', alpha=0.5)
            for j in range(lenMgFe):
                if G_MgFeFeH.data[0][i,j]>=Nlim:
                    ax.scatter(G_FeH.midpoints[0][i], G_MgFeFeH.integratedHist()[i,j], color='C1', alpha=0.5)
            ax.scatter(G_FeH.midpoints[0][i], (G_MgFeFeH.widths[1]*G_MgFeFeH.integratedHist()[i,:]).sum()/MgFeTotalWidth, color='C2', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('Integrated Mass per [Fe/H] per [Mg/Fe] /Msol dex^-2')
    
    fig, ax = plt.subplots()
    for i in range(lenFeH):
        if G_FeH.data[0][i]>=Nlim:
            ax.scatter(G_FeH.midpoints[0][i], 1/G_FeH.aR[i], color='C0', alpha=0.5)
            for j in range(lenMgFe):
                if G_MgFeFeH.data[0][i,j]>=Nlim:
                    ax.scatter(G_FeH.midpoints[0][i], 1/G_MgFeFeH.aR[i,j], color='C1', alpha=0.5)
    # ylim = ax.get_ylim()
    # ax.fill_between(G_FeH.midpoints[0], ylim[0], ylim[0]+(ylim[1]-ylim[0])*G_FeH.data[0]/max(G_FeH.data[0]), alpha=0.2, color='C2')
    ax.set_yscale('log')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('scale length /kpc')
    path = saveDir+'length'
    fig.savefig(path, dpi=300)
        
    fig, ax = plt.subplots()
    for i in range(lenFeH):
        if G_FeH.data[0][i]>=Nlim:
            ax.scatter(G_FeH.midpoints[0][i], 1/G_FeH.az[i], color='C0', alpha=0.5)
            for j in range(lenMgFe):
                if G_MgFeFeH.data[0][i,j]>=Nlim:
                    ax.scatter(G_FeH.midpoints[0][i], 1/G_MgFeFeH.az[i,j], color='C1', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('scale height /kpc')
    path = saveDir+'height'
    fig.savefig(path, dpi=300)

    
    


def plotageFeH():
    # Current smoothing breaks when whole FeH dist not included 
    FeHEdges = myUtils._FeH_edges_for_age
    ageEdges = myUtils._age_edges
    
    G = Galaxy.loadFromBins(labels=['FeH', 'age'], edges=[FeHEdges, ageEdges])
    
    Nlim = 100
    fig, axs = plt.subplots(ncols=5, figsize=[17, 4])
    titles = ['density at sun', 'integrated density', 'thick disk', 'aR', 'az']
    for i, im in enumerate([G.hist(), G.integratedHist(), G.hist(R=4,z=1), np.where(G.N>=Nlim, G.aR, 0), np.where(G.N>=Nlim, G.az, 0)]):
        axs[i].imshow(im.T, vmin=0, origin='lower', aspect='auto', extent=(FeHEdges[0], FeHEdges[-1], ageEdges[0], ageEdges[-1]))
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('[Fe/H]')
        axs[i].set_ylabel('age')
    fig.set_tight_layout(True)
    saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/analysis_FeHage_POLYDEG{POLYDEG}/'
    os.makedirs(saveDir, exist_ok=True)
    path = saveDir+'2Ds'
    fig.savefig(path, dpi=300)
    
    dists = [G.FeH('local'), G.FeH('MW', integrated=True), G.FeH('R2', R=2, z=0)]
    for d in dists:
        d.plot()
    dists[0].plotWith(dists[1])
    agename = ['young','mid,','old']
    
    # for i in range(3): # splitting bins
    #     agerange = ageEdges[i:i+2]
    #     print(agerange)
    #     G = Galaxy.loadFromBins(labels=['FeH', 'age'], edges=[FeHEdges, ageEdges[i:i+2]])
    #     dists = [G.FeH(f'local, {agename[i]}'), G.FeH(f'MW, {agename[i]}', integrated=True), G.FeH(f'R2, {agename[i]}', R=2, z=0)]
    #     for d in dists:
    #         d.plot()
    #     dists[0].plotWith(dists[1])
    

def plotFeH():
    FeHEdges = myUtils._FeH_edges
    G = Galaxy.loadFromBins(labels=['FeH'], edges=[FeHEdges])
    dists = [G.FeH('local'), G.FeH('MW', integrated=True), G.FeH('R2', R=2, z=0)]
    for d in dists:
        d.plot()
    dists[0].plotWith(dists[1])
    
    
def plotMgFeFeH():
    FeHEdges = myUtils._FeH_edges_for_MgFe
    MgFeEdges = myUtils._MgFe_edges
    # print(len(FeHEdges), len(MgFeEdges))
    labels = ['FeH', 'MgFe']
    edges = [FeHEdges,MgFeEdges]
    G = Galaxy.loadFromBins(labels, edges, noPyro=True)
    # print(G.hist())
    
    saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/analysis_FeHMgFe_POLYDEG{POLYDEG}/'
    os.makedirs(saveDir, exist_ok=True)

    
    fig, axs = plt.subplots(ncols=3, figsize=[18, 4])
    titles = ['density at sun', 'integrated density', 'R=4,z=1']
    for i, X in enumerate([G.hist(), G.integratedHist(), G.hist(R=4,z=1)]):
        image = axs[i].imshow(X.T, origin='lower', aspect='auto',
                              extent=(FeHEdges[0], FeHEdges[-1], MgFeEdges[0], MgFeEdges[-1]),
                              cmap=cmap1)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('[Fe/H]')
        axs[i].set_ylabel('[Mg/Fe]')
        fig.colorbar(image, ax=axs[i])
    fig.set_tight_layout(True)
    path = saveDir+'densities'
    fig.savefig(path, dpi=300)
    
    
    fig,ax = plt.subplots(figsize=[6, 4])
    image = ax.imshow(G.data[0].T, origin='lower', aspect='auto',
                      extent=(FeHEdges[0], FeHEdges[-1], MgFeEdges[0], MgFeEdges[-1]),
                      cmap=cmap1, norm=mpl.colors.LogNorm())
    ax.set_title('Stars per bin')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('[Mg/Fe]')
    fig.colorbar(image, ax=ax)
    fig.set_tight_layout(True)
    path = saveDir+'N'
    fig.savefig(path, dpi=300)
    
    Nlim = 50
    titles = ['scale length', 'scale height']
    fig, axs = plt.subplots(ncols=len(titles), figsize=[12, 4])
    for i, X in enumerate([np.where(G.data[0]>=Nlim, 1/G.aR, np.nan), np.where(G.data[0]>=Nlim, 1/G.az, np.nan)]):
        image = axs[i].imshow(X.T, origin='lower', aspect='auto',
                              extent=(FeHEdges[0], FeHEdges[-1], MgFeEdges[0], MgFeEdges[-1]),
                              cmap=mpl.colormaps['jet'], norm=mpl.colors.LogNorm())
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('[Fe/H]')
        axs[i].set_ylabel('[Mg/Fe]')
        fig.colorbar(image, ax=axs[i])
    fig.set_tight_layout(True)
    path = saveDir+'Bovy2012'
    fig.savefig(path, dpi=300)
    
    # dists = [G.FeH('local'), G.FeH('MW', integrated=True), G.FeH('R2', R=2, z=0)]
    
    # for d in dists:
    #     d.plot()
    # dists[0].plotWith(dists[1])
    
    
    
    


class Galaxy:
    def __init__(self, labels, edges, amp, aR, az, data):
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
        self.data = data
        
        self.shape = self.amp.shape
        self.widths = [(self.edges[i][1:] - self.edges[i][:-1]) for i in range(len(self.shape))]
        self.midpoints = [(self.edges[i][1:] + self.edges[i][:-1])/2 for i in range(len(self.shape))]
        self.vols = np.zeros(self.shape)
        for binNum in range(np.prod(self.shape)):
            multiIndex = np.unravel_index(binNum, self.shape)
            self.vols[multiIndex] = np.prod([self.widths[i][multiIndex[i]] for i in range(len(self.shape))])
        
    @classmethod
    def loadFromBins(cls, labels, edges, noPyro=False):
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
            else:
                with open(os.path.join(binDir, 'fit_results.dat'), 'rb') as f1:
                    logA, aR[multiIndex], az[multiIndex] = pickle.load(f1)
                
            with open(os.path.join(binDir, 'NRG2mass.dat'), 'rb') as f2:
                NRG2Mass = pickle.load(f2)
                
            amp[multiIndex] = NRG2Mass*np.exp(logA)/(np.prod(limits[:,1]-limits[:,0]))
        return cls(labels, edges, amp, aR, az, data)
    
    
    def hist(self, R=myUtils.R_Sun, z=myUtils.z_Sun, normalised=False):
        hist = self.amp*np.exp( - self.aR*(R-myUtils.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.hist(R,z)) #assumes bins cover whole distribution
    
    def integratedHist(self, Rlim=None, zlim=None, normalised=False):
        """integrates R=0 to R and z=-z to z, with default of whole galaxy"""
        hist = 4*np.pi*self.amp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az)
        if Rlim!=None:
            hist *= (1 - (1+self.aR*Rlim)*np.exp(-self.aR*Rlim))
        if zlim!=None:
            hist *= (1 - np.exp(-self.az*zlim))
            
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.integratedHist())
        
    def FeH(self, name, R=myUtils.R_Sun, z=myUtils.z_Sun, integrated=False, Rlim=None, zlim=None, normalised=False):
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
        return Distributions(FeHhist, FeHedges, name, self.binType, perVolume, normalised)



class Distributions:
    """Contains the binned FeH distributions, and methods to get corresponding smoothed and fH2O distributions.
    Also contiains data needed for plotting for ease
    Should normaliser be at later stage?"""
    
    alpha = 1
    
    def __init__(self, FeHhist, FeHedges, name, binType, perVolume, normalised=False):
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
        
        y = np.append(0, np.cumsum(self.FeHwidths*self.FeHhist))
        dist = scipy.interpolate.CubicSpline(self.FeHedges, y, bc_type='clamped', extrapolate=True).derivative()
        def FeHdistFunc(FeH):
            return np.where((self.FeHedges[0]<=FeH)&(FeH<self.FeHedges[-1]), dist(FeH), 0)
        self.FeHdist = FeHdistFunc
        
        def ISOsPerFeH(FeH):
            return self.alpha*(10**FeH)*self.FeHdist(FeH)
        
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
        return self.__class__(self.FeHhist, self.FeHedges, self.name, self. binType, self.perVolume, normalised=True)
        
    def plot(self):
        saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/analysis_{self.binType}_POLYDEG{POLYDEG}/'
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
        path = os.path.join(saveDir, self.name)
        fig.savefig(path, dpi=300)    
        
        

    def plotWith(self, dists2):
        assert self.binType==dists2.binType
        
        dists1 = self.butNormalised()
        dists2 = dists2.butNormalised()
        
        saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/analysis_{self.binType}_POLYDEG{POLYDEG}/'
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
        
        figname = f'{self.name} vs {dists2.name}'
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
        
# def SM2ISO(FeHdist, alpha=1, normalised=False):
#     def integrand(FeH):
#         return (10**FeH)*FeHdist(FeH)
    
#     normFactor = alpha*scipy.integrate.quad(integrand, -3, 3, limit=200)[0] if normalised else 1
    
#     def ISOdist(fH2O):
#         return -alpha*(10**compInv(fH2O))*FeHdist(compInv(fH2O))/(normFactor*compDeriv(compInv(fH2O)))
    
#     lowerEndCount = alpha*scipy.integrate.quad(integrand, FeHhigh, 3, limit=200)[0]/normFactor
#     upperEndCount = alpha*scipy.integrate.quad(integrand, -3, FeHlow, limit=200)[0]/normFactor
#     return (ISOdist, lowerEndCount, upperEndCount)



def binName(labels, limits):
    return '_'.join(['_'.join([labels[i], f'{limits[i][0]:.3f}', f'{limits[i][1]:.3f}']) for i in range(len(limits))])



if __name__=='__main__':
    main()