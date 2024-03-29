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

import astropy.table as aptable
import astropy.io 

import mySetup
# import myIsochrones
import myIsochrones2 as mI # has updates/improvements from Gaia projects


# For black holes paper


cmap0 = mpl.colormaps['Blues']
cmap01 = mpl.colormaps['Purples']
cmap1 = mpl.colormaps['Greys']#mpl.colormaps['Blues']
cmap2 = mpl.colormaps['hsv']
# colourPalette = mpl.colormaps['tab10'](np.linspace(0.05, 0.95, 10))
colours = ['darkseagreen', 'goldenrod', 'teal', 'crimson']

plt.rcParams.update({
    "text.usetex": True})
# plt.rcParams['font.family'] = 'serif'

# plotDir = r'D:\Moved folders\OneDrive\OneDrive\Documents\Code\APOGEE\plots'
# plotDir = f'/Users/hopkinsm/APOGEE/plots/BlackHoles/'
#f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/'
plotDir = '/home/hopkinsl/APOGEE/plots'
os.makedirs(plotDir, exist_ok=True)

# rBH = 1.48e-3 # calculated by hand, integrating Kroupa from 0.08 to inf vs 25 to inf
# rBH = 6.5e-3 integrating upwars of 8Msun, which is all remnants
rBH = mI.rBH
# TODO for revision - use updated isochrones, with upper limit to mass. Should be mI.rBH() = 1.34e-3


def main():
    G = Galaxy.loadFromBins()
    
    # plotFit(G, G.mask())
    # totalBHs(G)
    # ageOverR(G)
    # plotOverR(G)
    # plotmidplane(G)
    # plotObs(G)
    # print(PISSfraction(G))
    print(F1(G))
    
    # p = KSdiff(G, 95)
    # fig, ax = plt.subplots()
    # ax.hist(p, bins=np.linspace(0,0.5, 21))#p curve
    # ax.axvline(0.05, ls='--', c='k')
    # print(np.count_nonzero(p<0.05)/len(p), p.mean(), p.std())
    
    

def dataForHeloise():
    G = Galaxy.loadFromBins()
    
    ageWidth = 1 # technically I'm not sampling age distribution great here, but fine
    mhWidth = 0.1
    RWidth = 0.2
    mh_mid = np.round(np.arange(15)*mhWidth-0.95, 2)
    age_mid = np.arange(14)*ageWidth+0.5
    R_mid = np.round(np.arange(40)*RWidth+4.1, 1)
    print(age_mid)
    
    print(mh_mid,R_mid,age_mid)
    table = aptable.Table(np.array(np.meshgrid(R_mid,mh_mid,age_mid)).reshape(3,-1).T, names=['R', 'FeH', 'age'])
    
    RGcol = np.zeros(len(table))
    SMcol = np.zeros(len(table))
    
    
    def smDist(R,mh,age):
        ageF = np.sqrt(G.omega/(2*np.pi))*np.exp(-0.5*G.omega*(age-G.tau0)**2) 
        # grid of age factors at age for each FeH aFe combo
        # print(ageF)
        FeHvals = (G.FeH(G.zintegratedHist(R)*ageF, G.mask())*G.FeHWidths)*ageWidth
        # val of dist at R and age for each FeH
        # print(FeHvals)
        return FeHvals[np.digitize(mh, G.FeHEdges)-1]
        
    def rgDist(R,mh,age):
        ageF = np.sqrt(G.omega/(2*np.pi))*np.exp(-0.5*G.omega*(age-G.tau0)**2)*ageWidth
        # grid of age factors at age for each FeH aFe combo
        # print(ageF)
        FeHvals = (G.FeH(G.zintegratedHist(R)*ageF/G.NRG2NSM, G.mask())*G.FeHWidths)*ageWidth
        # val of dist at R and age for each FeH
        # print(FeHvals)
        return FeHvals[np.digitize(mh, G.FeHEdges)-1]
        
    # I multiply by FeWidth and ageWidth but not Rwidth
    # so that result is surface density of stars in that R,age,FeH bin
    i = 0
    for row in table.iterrows():
        # R,mh,age = row
        # dist = fullDist(R,mh,age)
        SMcol[i] = smDist(*row)
        RGcol[i] = rgDist(*row)
        i+=1
    
    SMColumn = aptable.Column(name='SM', data=SMcol)
    RGColumn = aptable.Column(name='RG', data=RGcol)
    table.add_columns((RGColumn,SMColumn))
    table.write('/home/hopkinsl/APOGEE/table.txt', format='ascii', overwrite=True)
    
def testHeloise():
    table = astropy.io.ascii.read('/home/hopkinsl/APOGEE/table.txt')
    print(table)
    Rvals = np.unique(table['R'])
    FeHvals = np.unique(table['FeH'])
    agevals = np.unique(table['age'])
    
    mask = (table['age']==0.5)&(table['FeH']==0.05)
    fig,ax=plt.subplots()
    ax.plot(table['R'][mask], table['RG'][mask], label='RG')
    ax.plot(table['R'][mask], table['SM'][mask]/1000, label='SM/1000')
    ax.set_xlabel('R / kpc')
    ax.set_ylabel('Surface Density')
    ax.set_title('[Fe/H]=0.05, age=0.5Gyr')
    ax.legend()
    
    
    FeHmask = (table['FeH']==-0.45)
    RGsums = np.zeros(len(Rvals))
    SMsums = np.zeros(len(Rvals))
    for i in range(len(Rvals)):
        mask = FeHmask&(table['R']==Rvals[i])
        RGsums[i] = np.sum(table[mask]['RG'])
        SMsums[i] = np.sum(table[mask]['SM'])
    fig,ax=plt.subplots()
    ax.plot(Rvals, RGsums, label='RG')
    ax.plot(Rvals, SMsums/1000, label='SM/1000')
    ax.set_xlabel('R / kpc')
    ax.set_ylabel('Surface Density')
    ax.set_title('[Fe/H]=-0.45, all ages')
    ax.legend()
    
    
    RGsums = np.zeros(len(Rvals))
    SMsums = np.zeros(len(Rvals))
    for i in range(len(Rvals)):
        R_mask = (table['R']==Rvals[i])
        RGsums[i] = np.sum(table[R_mask]['RG'])
        SMsums[i] = np.sum(table[R_mask]['SM'])
    fig,ax=plt.subplots()
    ax.plot(Rvals, RGsums, label='RG')
    ax.plot(Rvals, SMsums/1000, label='SM/1000')
    ax.set_xlabel('R / kpc')
    ax.set_ylabel('Surface Density')
    ax.set_title('all FeH, all ages')
    ax.legend()
    
    
    RGsums = np.zeros(len(agevals))
    SMsums = np.zeros(len(agevals))
    for i in range(len(agevals)):
        mask = (table['age']==agevals[i])
        RGsums[i] = np.sum(table[mask]['RG']*table[mask]['R'])
        SMsums[i] = np.sum(table[mask]['SM']*table[mask]['R']) # multiplication by R because integrating over an annulus
    fig,ax=plt.subplots()
    ax.plot(agevals, RGsums/RGsums.sum(), label='RG')
    ax.plot(agevals, SMsums/SMsums.sum(), label='SM')
    ax.set_xlabel('age / Gyr')
    ax.set_ylabel('normalised age distribution / 1/Gyr')
    ax.set_title('all FeH, all R')
    ax.legend()
    
    
def totalBHs(G):
    rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    # starcounts = G.FeH(G.integratedHist(Rlim1=0), G.mask())*G.FeHWidths 
    starcounts = G.FeH(G.hist(), G.mask())*G.FeHWidths # aroudnSun - per kpc3?
    
    BHcount = (rBHarray*starcounts).sum()
    # BHcount = (rBH*starcounts).sum() for no metallicity cutoff
    
    
    smCount = starcounts.sum()
    print(BHcount, smCount, BHcount/smCount)
    

    
def maxCentreVals(G):
    cv = G.logNuSun + G.aR*mySetup.R_Sun# + np.log(G.NRG2NLS)
    cv_masked = cv[G.mask()]
    inOrder = np.sort(cv_masked)
    print(inOrder)
    
    FeHindx, aFeindx = np.where(cv>=inOrder[-3]) # should fetch highest n - plus these outside mask
    
    return (G.FeHMidpoints[FeHindx], G.aFeMidpoints[aFeindx])


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
    
    isogrid = mI.loadGrid()
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
        
        rg2sm[i] = mI.NRG2NSM(isochrones, meanTau0[i], 1)
        rg2ls[i] = mI.NRG2NLS(isochrones, meanTau0[i], 1)
    
    overage_rg2sm = np.zeros(len(tau))
    overage_rg2ls = np.zeros(len(tau))
    for j in range(len(tau)):
        overage_rg2sm[j] = mI.NRG2NSM(isochrones, tau[j], 1)
        overage_rg2ls[j] = mI.NRG2NLS(isochrones, tau[j], 1)
        
    fig, ax = plt.subplots()
    ax.plot(tau, overage_rg2sm)
    ax.plot(tau, overage_rg2ls)
    ax.set_xlabel(r'age / Gyr')
    ax.set_ylabel(r'rg2_')
    # path = os.path.join(plotDir, '.pdf')
    # fig.tight_layout()
    # fig.savefig(path, dpi=100)
    
    fig, ax = plt.subplots()
    ax.plot(R, meanTau0)
    ax.set_xlabel(r'$R\;/\;\mathrm{kpc}$')
    ax.set_ylabel(r'mean tau0 / Gyr')
    path = os.path.join(plotDir, 'tau0overR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    
    fig,ax = plt.subplots()
    ax.imshow(ageDist.T, origin='lower',
              extent=(R[0], R[-1], tau[0], tau[-1]),
              aspect='auto')
    
    fig, ax = plt.subplots()
    ax.plot(R, SMstars/RGstars, label='full measured')
    ax.plot(R, rg2sm, label='estimated from mean age')
    ax.set_xlabel(r'$R\;/\;\mathrm{kpc}$')
    ax.set_ylabel(r'rg2sm')
    fig.legend()
    path = os.path.join(plotDir, 'rg2smOverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    
    fig, ax = plt.subplots()
    ax.plot(R, LSstars/RGstars, label='full measured')
    ax.plot(R, rg2ls, label='estimated from mean age')
    ax.set_xlabel(r'$R\;/\;\mathrm{kpc}$')
    ax.set_ylabel(r'rg2ls')
    fig.legend()
    path = os.path.join(plotDir, 'rg2lsOverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    
    fig, ax = plt.subplots()
    ax.plot(R, SMstars/LSstars, label='full measured')
    ax.plot(R, rg2sm/rg2ls, label='estimated from mean age')
    ax.set_xlabel(r'$R\;/\;\mathrm{kpc}$')
    ax.set_ylabel(r'ls2sm')
    fig.legend()
    path = os.path.join(plotDir, 'ls2smOverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    
    print(rg2ls/rg2sm)
    

def F1(G):
    """Integrates midplane densities weighted by x*(1-x) optical depth factor
    """
    
    N = 101
    R = np.linspace(0,mySetup.R_Sun,N)
    R = (R[:-1]+R[1:])/2
    
    # rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    
    volSM = np.zeros(len(R))
    volLS = np.zeros(len(R))
    volBH = np.zeros(len(R))
    
    for i in range(len(R)):
        volSM[i], volLS[i], volBH[i] = countsAtR(G,R[i], volume=True)    
        
    x = np.linspace(0,1,N)
    x = (x[:-1]+x[1:])/2
    
    averageBHmass = 20
    
    averageLensMass = 0.4
    BDperSM = 0.78 # brown dwarfs per sine morte star
    # both calculated by hand
    
    # F1 = (averageBHmass/mI.meanMini)*(np.sum(volBH*x*(1-x))/np.sum(volLS*x*(1-x)))
    F1 = (averageBHmass*np.sum(volBH*x*(1-x)))/(averageLensMass*(np.sum(volLS*x*(1-x)) + BDperSM*np.sum(volSM*x*(1-x))))

    
    return F1
    

def countsAtR(G,R, volume=False, co=0):
    #co is cutoff
    rBHarray = np.where(G.FeHMidpoints<co, rBH, 0.0)
    
    if volume:
        SMFeHcounts = (G.FeH(G.hist(R), G.mask())*G.FeHWidths)
        LSFeHcounts = (G.FeH(G.hist(R)*G.NRG2NLS/G.NRG2NSM, G.mask())*G.FeHWidths)
    else:
        SMFeHcounts = (G.FeH(G.zintegratedHist(R), G.mask())*G.FeHWidths)
        LSFeHcounts = (G.FeH(G.zintegratedHist(R)*G.NRG2NLS/G.NRG2NSM, G.mask())*G.FeHWidths)
        
    SM = SMFeHcounts.sum() 
    LS = LSFeHcounts.sum() 
    BH = (SMFeHcounts*rBHarray).sum()
    
    return SM,LS,BH


def PISSfraction(G):
    # look at gaia version of myIsochrones, should integrate analytically and use an upper bound
    counts = (G.FeH(G.integratedHist(), G.mask())*G.FeHWidths)
    
    FeHfrac = np.sum(counts[G.FeHMidpoints<-1.3])/np.sum(counts)
    
    return (FeHfrac)
    
    
    

def plotmidplane(G, co=0.0):
    
    R = np.linspace(0,mySetup.R_Sun,101)
    R = (R[:-1]+R[1:])/2
    
    # rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    
    # surSM = np.zeros(len(R))
    # surLS = np.zeros(len(R))
    # surBH = np.zeros(len(R))
    volSM = np.zeros(len(R))
    volLS = np.zeros(len(R))
    volBH = np.zeros(len(R))
    
    for i in range(len(R)):
        # surSM[i], surLS[i], surBH[i] = countsAtR(G,R[i])
        volSM[i], volLS[i], volBH[i] = countsAtR(G,R[i], volume=True, co=co)
        
    
    # LSadjusted_volBH = volLS*surBH/surLS
    # SMadjusted_volBH = volSM*surBH/surSM
        
    fig, axs = plt.subplots(nrows=2, sharex=True)
    fig.set_figheight(4.0*1.2)
    ax=axs[0]
    ax.plot(R[R>3.99], volLS[R>3.99]*1e-9, color=colours[1], label='Living Stars')
    ax.plot(R[R<4.01], volLS[R<4.01]*1e-9, linestyle='dashed', color=colours[1])
    ax.plot(R[R>3.99], volBH[R>3.99]*1e3*1e-9, color=colours[2], label=r'Black Holes$\,\times 10^3$')
    ax.plot(R[R<4.01], volBH[R<4.01]*1e3*1e-9, linestyle='dashed', color=colours[2])
    ax.set_ylabel(r'Volume Density /$\;\mathrm{pc}^{-3}$')
    ax.legend()
    
    ax=axs[1]
    ax.plot(R[R>3.99], volBH[R>3.99]*1e3/volLS[R>3.99], color=colours[3], label='Black Holes / Living Stars')
    ax.plot(R[R<4.01], volBH[R<4.01]*1e3/volLS[R<4.01], color=colours[3], linestyle='dashed')
    ax.set_xlabel(r'$R\;/\;\mathrm{kpc}$')
    ax.set_ylabel(r'Volume Density Ratio $\times10^3$')
    ax.set_ylim([0, 1.0])
    ax.legend()
    path = os.path.join(plotDir, 'midoverR.pdf')
    fig.subplots_adjust(hspace=0.1) #makes them adjacent
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    

def plotObs(G, co=0.0):
    
    R = np.linspace(0,mySetup.R_Sun,101)
    R = (R[:-1]+R[1:])/2
    dR = R[1]-R[0]
    
    volSM = np.zeros(len(R))
    volLS = np.zeros(len(R))
    volBH = np.zeros(len(R))
    
    for i in range(len(R)):
        volSM[i], volLS[i], volBH[i] = countsAtR(G,R[i], volume=True, co=co)
    
    
    volSM *= R*(mySetup.R_Sun-R)
    volLS *= R*(mySetup.R_Sun-R)
    volBH *= R*(mySetup.R_Sun-R)
    
    volSM /= volSM.sum()*dR
    volLS /= volLS.sum()*dR
    volBH /= volBH.sum()*dR
        
    
    fig, ax = plt.subplots()
    ax.plot(R, volLS, color=colours[1], label='Living Stars')
    ax.plot(R, volBH, color=colours[2], label=r'Black Holes')
    ax.set_xlabel(r'$R\;/\;\mathrm{kpc}$')
    ax.set_ylabel(r'Observable Distribution /$\;\mathrm{kpc}^{-1}$')
    # ax.set_yscale('log')
    ax.legend()
    path = os.path.join(plotDir, 'obsoverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    
    # fig, ax = plt.subplots()
    # ax.plot(R[R>3.99], volBH[R>3.99]/volLS[R>3.99], color=colours[3], label='Black holes / living stars')
    # ax.plot(R[R<4.01], volBH[R<4.01]/volLS[R<4.01], color=colours[3], linestyle='dashed')
    # ax.set_xlabel(r'$R/\mathrm{kpc}$')
    # ax.set_ylabel(r'Midplane volume density ratio')
    # ax.set_ylim([0, 1.0e-3])
    # ax.ticklabel_format(axis='y',scilimits=(0,0))
    # ax.legend()
    # path = os.path.join(plotDir, 'obsratiosoverR.pdf')
    # fig.tight_layout()
    # fig.savefig(path, dpi=100)
    # print(SMstars/LSstars)
    
        
        
def plotOverR(G):
    R = np.linspace(4,12,101)
    R = (R[:-1]+R[1:])/2
        
    surSM = np.zeros(len(R))
    surLS = np.zeros(len(R))
    surBH = np.zeros(len(R))
    
    for i in range(len(R)):
        surSM[i], surLS[i], surBH[i] = countsAtR(G,R[i])
        
        
    #mpl default figsize is 6.0 for the width and 4.0 for hight (inches), corresponding to width of a page. Try halfing these for figures I'll put side by side
    #could try putting both on same plot I'f I'm presenting them like that - do need to adjust aspect ratio
    #however best thing is probably shared x-axis
    fig, axs = plt.subplots(nrows=2, sharex=True)
    fig.set_figheight(4.0*1.2)
    ax = axs[0]
    ax.plot(R[R>3.95], surSM[R>3.95]*1e-6, color=colours[0], label='Sine Morte Stars')
    # ax.plot(R[R<4], surSM[R<4], '--C0')
    ax.plot(R[R>3.95], surLS[R>3.95]*1e-6, color=colours[1], label='Living Stars')
    # ax.plot(R[R<4], surLS[R<4]*1e-6, '--C2')
    ax.plot(R[R>3.95], surBH[R>3.95]*1e3*1e-6, color=colours[2], label=r'Black Holes$\,\times 10^3$')
    # ax.plot(R[R<4], surBH[R<4]*1e3*1e-6, '--C1')
    ax.set_ylabel(r'Surface Density /$\;\mathrm{pc}^{-2}$')
    ax.legend()
    
    ax = axs[1]
    ax.plot(R[R>3.95], surBH[R>3.95]*1e3/(surLS[R>3.95]), color=colours[3], label='Black Holes / Living Stars')
    # ax.plot(R[R<4], surBH[R<4]*1e3/surLS[R<4], '--C3')
    ax.set_xlabel(r'$R\;/\;\mathrm{kpc}$')
    ax.set_ylabel(r'Surface Density Ratio $\times 10^3$')
    ax.set_ylim([0, 1.5])
    ax.legend()
    path = os.path.join(plotDir, 'overR.pdf')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1) #makes them adjacent
    fig.savefig(path, dpi=100)
    print(fig.get_size_inches())
    # print(surSM/surLS)
    
    
def plotTriangle(G):
    """bulge fields extend down to -8 degrees latitude, a kpc away from midplane
    So lensing actually depends density over a triangle up to """
    R = np.linspace(0,mySetup.R_Sun,51)
    R = (R[:-1]+R[1:])/2
    
    z = (mySetup.R_Sun-R)*np.tan(8*np.pi/180)
    print(z)
    
    rBHarray = np.where(G.FeHMidpoints<0.0, rBH, 0.0)
    
    SMstars = np.zeros(len(R))
    LSstars = np.zeros(len(R))
    BHs = np.zeros(len(R))
    
    for i, r in enumerate(R):
        SMstarFeHcounts = (G.FeH(G.zintegratedHist(r, z[i]), G.mask())*G.FeHWidths)
        LSstarFeHcounts = (G.FeH(G.zintegratedHist(r, z[i])*G.NRG2NLS/G.NRG2NSM, G.mask())*G.FeHWidths)
        SMstars[i] = SMstarFeHcounts.sum()/z[i]
        LSstars[i] = LSstarFeHcounts.sum()/z[i]
        BHs[i] = (SMstarFeHcounts*rBHarray).sum()/z[i]
        
        
    fig, ax = plt.subplots()
    ax.plot(R[R>3.95], LSstars[R>3.95]*1e-9, 'C2', label='Living stars')
    ax.plot(R[R<4],    LSstars[R<4]   *1e-9, '--C2')
    ax.plot(R[R>3.95], SMstars[R>3.95]*1e-9,'C0', label='Sine morte stars')
    ax.plot(R[R<4],    SMstars[R<4]   *1e-9, '--C0')
    ax.plot(R[R>3.95], BHs[R>3.95]*1e3*1e-9, 'k', label=r'Black holes$\,\cdot 10^3$')
    ax.plot(R[R<4],    BHs[R<4]   *1e3*1e-9, '--k')
    ax.set_xlabel(r'$R\;/\;\mathrm{kpc}$')
    ax.set_ylabel(r'Average volume density /$\;\mathrm{number}\;\mathrm{pc}^{-3}$')
    # ax.set_yscale('log')
    ax.legend()
    path = os.path.join(plotDir, 'triangleoverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    
    fig, ax = plt.subplots()
    ax.plot(R[R>3.95], BHs[R>3.95]/LSstars[R>3.95], 'C3', label='Black holes / living stars')
    ax.plot(R[R<4], BHs[R<4]/LSstars[R<4], '--C3')
    # ax.plot(R[R>3.95], SMstars[R>3.95]/LSstars[R>3.95], 'C1', label=r'SM / living stars')
    ax.scatter(12,0.5, color='white')#makes x axis same as midplane
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'average volume density ratio')
    ax.set_ylim([0, 1.5e-3])
    ax.ticklabel_format(axis='y',scilimits=(0,0))
    ax.legend()
    path = os.path.join(plotDir, 'triangleratiosoverR.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print(SMstars/LSstars)
    
    
def KSdiff(G,N, co=0.0):
    """ finds how many draws from BH midplane dist needed to get 5% KS test to stellar dist
    
    Note R is diffed here so values between posts, taken as average when integrated between posts.
    LSstars and BHs are values between posts, LSdist interpolates, 
    so cumulative sums are evalued at posts using estimations between posts
    first and last post assumed 0 and 1
    
    23/09/23 - detection probability has additional R(R_0 - R) selection effect
    """
    
    undiffedR = np.linspace(0,mySetup.R_Sun,51)
    R = (undiffedR[:-1]+undiffedR[1:])/2
    dR= R[1]-R[0]
    
    
    volSM = np.zeros(len(R))
    volLS = np.zeros(len(R))
    volBH = np.zeros(len(R))
    
    for i in range(len(R)):
        volSM[i], volLS[i], volBH[i] = countsAtR(G,R[i], volume=True, co=co)
    
    volSM *= R*(mySetup.R_Sun-R)
    volLS *= R*(mySetup.R_Sun-R)
    volBH *= R*(mySetup.R_Sun-R)
    
    BHcs = np.append([0], np.cumsum(volBH))
    BHcs/=BHcs[-1]
    LScs = np.append([0], np.cumsum(volLS))
    LScs/=LScs[-1]
    # print(LScs)
    
    LSdist = lambda r: np.interp(r, undiffedR, LScs)
    # since LScs is cumulative sum up to that R, -dR/2 makes distirbution
    #oscillate about correct, not under correct
    
    pR = np.linspace(0, mySetup.R_Sun, 100)
    fig, ax = plt.subplots()
    ax.plot(pR, LSdist(pR))
    ax.plot(undiffedR, LScs)
    ax.plot(undiffedR, BHcs)
    # ax.set_title(f'cutoff: {co}')
    fig, ax = plt.subplots()
    ax.plot((pR[:-1]+pR[1:])/2, (LSdist(pR[1:])-LSdist(pR[:-1]))/(pR[1]-pR[0]))
    ax.plot(R, (LScs[1:]-LScs[:-1])/dR)
    ax.plot(R, (BHcs[1:]-BHcs[:-1])/dR)
    # ax.set_title(f'cutoff: {co}')
    
    p = np.array([])
    
    for i in range(100): #5000 works well
        draws = np.random.uniform(size=N)
        BHRs = R[np.digitize(draws,BHcs)-1]
        p = np.append(p, scipy.stats.kstest(BHRs, LSdist).pvalue)
    
    print(N)
    return p

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
        fig.savefig(path, dpi=100)




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
        
        isogrid = mI.loadGrid()
        
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
                        NRG2NSM[i,j] = mI.NRG2NSM(isochrones, 12, 1) #uses old age to get upper lim 
                        NRG2NLS[i,j] = mI.NRG2NLS(isochrones, 12, 1) #uses old age to get upper lim 
                    else:
                        NRG2NSM[i,j] = mI.NRG2NSM(isochrones, tau0[i,j], omega[i,j]) 
                        NRG2NLS[i,j] = mI.NRG2NLS(isochrones, tau0[i,j], omega[i,j]) 
                    
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
    # main()
    testHeloise()
    
    
    
    
    