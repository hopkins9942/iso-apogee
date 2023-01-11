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

SMOOTH_FeH=True
#FINE=True
POLYDEG=3
PROPZ=True

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:7]
print(repo)
print(sha)


saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/FeHanalysis/{POLYDEG}-{SMOOTH_FeH}-{PROPZ}/'

binsDir = '/Users/hopkinsm/data/APOGEE/bins'


def main():
    
    _FeH_edges = myUtils.arr((-1.975, 0.725, 0.1)) #0.625-09.725 has no APOGEE statsample stars, -1.575--1.475 has about 130
    binList = [{'FeH': (_FeH_edges[i], _FeH_edges[i+1])} for i in range(len(_FeH_edges)-1)]
    
    FeHedges = np.append([binList[i]['FeH'][0] for i in range(len(binList))],
                         binList[-1]['FeH'][1])
    FeHmidpoints = (FeHedges[:-1] + FeHedges[1:])/2
    FeHwidths = FeHedges[1:] - FeHedges[:-1]
    
    FeHnumberlogA  = np.zeros(len(binList))
    aR       = np.zeros(len(binList))
    az       = np.zeros(len(binList))
    NRG2mass = np.zeros(len(binList))
    
    for i, binDict in enumerate(binList):
        FeHnumberlogA[i], aR[i], az[i] = loadFitResults(binDict)
        if (FeHnumberlogA[i]==-999):
            # no stars in bin, fit not attempted
            aR[i], az[i] = 0.000001, 0.0000001
            NRG2mass[i] = 0
        else:
            NRG2mass[i] = loadNRG2mass(binDict)
        
    #NRGDM = densityModel(FeHEdges, np.exp(FeHnumberlogA)/FeHWidths, aR, az)
    StellarMassDM = densityModel(FeHedges, NRG2mass*np.exp(FeHnumberlogA)/FeHwidths, aR, az)
    
    print('Check hist')
    print(StellarMassDM.hist())

    
    
    
    
    #plotting
    FeH_plot = np.linspace(-1.5, 0.9, 15*20+1)
    fH2O_plot = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
    
    
    fig, ax = plt.subplots()
    ax.plot(FeH_plot, comp(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,0.1], color=f'C{i}')
    # ax.set_xlim([-1,0.5])
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel(r'$f_\mathrm{H_2O}$')
    saveFig(fig,'comp.png')
    
    fig, ax = plt.subplots()
    ax.plot(FeH_plot, compDeriv(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,-0.1], color=f'C{i}')
    # ax.set_xlim([-1,0.5])
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel(r'$\frac{\mathrm{d}f_\mathrm{H_2O}}{\mathrm{d[Fe/H]}}$')
    saveFig(fig,'compDeriv.png')
    
    fig, ax = plt.subplots()
    ax.plot(fH2O_plot, compInv(fH2O_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([comp(FeH),comp(FeH)], [0,0.1], color=f'C{i}') #colours correspond on multiple graphs
    ax.set_xlabel(r'fH2O')
    saveFig(fig, 'compInv.png')
    
    
    
    # # EAGLE
    EAGLE_data = np.loadtxt('/Users/hopkinsm/data/APOGEE/input_data/EAGLE_MW_L0025N0376_REFERENCE_ApogeeRun_30kpc_working.dat') 
    # List of star particles
    EAGLE_mass = EAGLE_data[:,9]
    EAGLE_FeH = EAGLE_data[:,14]
    EAGLEedges = myUtils.arr((-2.975, 1.025, 0.1))
    EAGLEwidths = EAGLEedges[1:] - EAGLEedges[:-1]
    FeHhist_EAGLE = np.histogram(EAGLE_FeH, bins=EAGLEedges, weights=EAGLE_mass/EAGLEwidths[0])[0]
    print(FeHhist_EAGLE)
    
    APOedges = np.append([binList[i]['FeH'][0] for i in range(len(binList))],
                         binList[-1]['FeH'][1])
    
    namesList = ['Solar Neighbourhood', 'GalCentre', 'Milky Way', 'EAGLE', 'R=2kpc', 'within2', 'outside2']
    FeHedgesList = [APOedges, APOedges, APOedges, EAGLEedges, APOedges, APOedges, APOedges]
    FeHhistsList = [StellarMassDM.hist(),
                    StellarMassDM.hist((0,0)),
                    StellarMassDM.integratedHist(),
                    FeHhist_EAGLE,
                    StellarMassDM.hist((2,0)),
                    StellarMassDM.histWithin(2),
                    StellarMassDM.integratedHist()-StellarMassDM.histWithin(2)
                    ]
    perVolume = [True, True, False, False, True, False, False]
    
    for plotNum in range(len(namesList)): # to save repeating similar code
        name = namesList[plotNum]
        FeHedges = FeHedgesList[plotNum]
        FeHmidpoints = (FeHedges[:-1] + FeHedges[1:])/2
        FeHwidths = FeHedges[1:] - FeHedges[:-1]
        
        FeHhist = FeHhistsList[plotNum]
        FeHplotPoints = np.linspace(FeHedges[0], FeHedges[-1], 10*len(FeHwidths))
        print(FeHhist)
        FeHdist = hist2dist(FeHedges, FeHhist)
        
        fH2OplotPoints = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
        fH2Odist, lowerCount, upperCount = SM2ISO(FeHdist)
        middleCount = quad(fH2Odist, fH2OplotPoints[0], fH2OplotPoints[-1])[0]
        #LBMHighCount = quad(fH2Odist, 0.4, fH2OplotPoints[-1])[0] +upperCount
        
        FeHunit = (r'$\mathrm{M}_\odot \mathrm{dex}^{-1} \mathrm{kpc}^{-3}$' if perVolume[plotNum] 
                   else r'$\mathrm{M}_\odot \mathrm{dex}^{-1}$')
        fH2Ounit = (r'$\mathrm{ISOs} \; \mathrm{kpc}^{-3}$' if perVolume[plotNum] 
                    else r'$\mathrm{ISOs}$')
        fH2OintUnit = (r'$\mathrm{ISOs} \; \mathrm{kpc}^{-3}$' if perVolume[plotNum] 
                    else r'$\mathrm{ISOs}$')
        
        fH2Oheight = fH2Odist(0.3)*4
        
        
        fig, axs = plt.subplots(ncols=3, figsize=[12, 4])
        axs[0].bar(FeHmidpoints, FeHhist, width = FeHwidths, alpha=0.5)
        axs[0].plot(FeHplotPoints, FeHdist(FeHplotPoints), color='C1')
        axs[0].plot([-0.4,-0.4], [0,FeHdist(0)], color='C2', alpha=0.5)
        axs[0].plot([ 0.4, 0.4], [0,FeHdist(0)], color='C2', alpha=0.5)
        # for i, FeH in enumerate(FeHedges):
        #     ax.plot([FeH,FeH], [0,FeHdist(0)/10], color=f'C{i}')
        axs[0].set_xlabel(r'[Fe/H]')
        axs[0].set_ylabel(f'Stellar mass distribution ({FeHunit})')
        #axs[0].set_title(name)
        
        print(f'Check spline method: {plotNum}')
        print(np.array([quad(FeHdist, FeHedges[i], FeHedges[i+1])[0]/FeHwidths[i]
                            for i in range(len(FeHmidpoints))]) - FeHhist)
        
        #fig, ax = plt.subplots(ncols=2)
        axs[1].plot(fH2OplotPoints, fH2Odist(fH2OplotPoints))
        # ax.text(fH2Olow, fH2Oheight*0.9, f"lower: {lowerCount:.2e} {fH2OintUnit}")
        # ax.text(fH2Olow, fH2Oheight*0.8, f"middle: {middleCount:.2e} {fH2OintUnit}")
        # ax.text(fH2Olow, fH2Oheight*0.7, f"upper: {upperCount:.2e} {fH2OintUnit}")
        # for i, FeH in enumerate(FeHedges):
        #     ax[0].plot([comp(FeH),comp(FeH)], [0,fH2Oheight/10], color=f'C{i}') #colours correspond on multiple graphs
        # ax[0].set_ylim(0,fH2Oheight)
        axs[1].set_ylim(bottom=0)
        axs[1].set_xlabel(r'$f_\mathrm{H_2O}$')
        axs[1].set_ylabel(f'ISO distribution ({fH2Ounit})')
        axs[2].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  [lowerCount, middleCount, upperCount])
        axs[2].set_ylabel(f'ISO distribution ({fH2OintUnit})')
        fig.suptitle(f'{name}')
        fig.set_tight_layout(True)
        if plotNum in [0,2,3,4]:
            saveFig(fig, f'{name}.png')
        
    # comparisons
    comparisonIndices = [(0,2), (2,3), (0,1), (0,4)] #Integrated vs solar neighborhood, Integrated vs EAGLE
    for count in range(len(comparisonIndices)):
        p1, p2 = comparisonIndices[count]
        FeHedges1 = FeHedgesList[p1]
        FeHedges2 = FeHedgesList[p2]
        
        FeHplotPoints = (np.linspace(APOedges[0], APOedges[-1], 10*(len(APOedges)-1)) if count!=1
                         else np.linspace(EAGLEedges[0], EAGLEedges[-1], 10*(len(EAGLEedges)-1)))
        FeHdist1 = hist2dist(FeHedges1, FeHhistsList[p1], normalised=True)
        FeHdist2 = hist2dist(FeHedges2, FeHhistsList[p2], normalised=True)
        name = namesList[p1]+' vs '+namesList[p2]
        
        fig, axs = plt.subplots(ncols=3, figsize=[12, 4])
        axs[0].plot(FeHplotPoints, FeHdist1(FeHplotPoints), label=namesList[p1])
        axs[0].plot(FeHplotPoints, FeHdist2(FeHplotPoints), label=namesList[p2])
        axs[0].plot([-0.4,-0.4], [0,FeHdist1(0)], color='C2', alpha=0.5)
        axs[0].plot([ 0.4, 0.4], [0,FeHdist1(0)], color='C2', alpha=0.5)
        axs[0].legend()
        axs[0].set_xlabel(r'[Fe/H]')
        axs[0].set_ylabel(r'Normalised SM distribution ($\mathrm{dex}^{-1}$)')
        #axs[0].set_title(name)
        
        fH2OplotPoints = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
        fH2Odist1, lowerCount1, upperCount1 = SM2ISO(FeHdist1, normalised=True)
        middleCount1 = quad(fH2Odist1, fH2OplotPoints[0], fH2OplotPoints[-1])[0]
        fH2Odist2, lowerCount2, upperCount2 = SM2ISO(FeHdist2, normalised=True)
        middleCount2 = quad(fH2Odist2, fH2OplotPoints[0], fH2OplotPoints[-1])[0]
        
        fH2Oheight = max(fH2Odist1(0.3)*4, fH2Odist2(0.3)*4)
        
        #fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        axs[1].plot(fH2OplotPoints, fH2Odist1(fH2OplotPoints), label=namesList[p1])
        axs[1].plot(fH2OplotPoints, fH2Odist2(fH2OplotPoints), label=namesList[p2])
        # ax[0].set_ylim(0,fH2Oheight)
        axs[1].set_ylim(bottom=0)
        axs[1].set_xlabel(r'$f_\mathrm{H_2O}$')
        axs[1].set_ylabel('Normalised ISO distribution')
        axs[1].legend()
        axs[2].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  [lowerCount1, middleCount1, upperCount1], alpha=0.5, label=namesList[p1])
        axs[2].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  [lowerCount2, middleCount2, upperCount2], alpha=0.5, label=namesList[p2])
        axs[2].set_ylabel('Normalised ISO distribution')
        axs[2].legend()
        fig.suptitle(f'{name}')
        fig.set_tight_layout(True)
        saveFig(fig, f'{name}.png')
    
        
    # dist vs R
    R = np.linspace(0,20,100)
    R = (R[:-1]+R[1:])/2
    FeH = np.linspace(-1,0.5,100)
    FeH = (FeH[:-1] + FeH[1:])/2
    fH2O = np.linspace(0.1,0.5,100)
    fH2O = (fH2O[:-1] + fH2O[1:])/2
    
    spatialFeHDist = np.zeros((len(FeH),len(R)))
    densityDist = np.zeros(len(R))
    spatialfH2ODist = np.zeros((len(fH2O),len(R)))
    
    for i, r in enumerate(R):
        densityDist[i] = (APOedges[1]-APOedges[0])*np.sum(StellarMassDM.hist(position=(r,0)))
        FeHdist = hist2dist(APOedges, StellarMassDM.hist(position=(r,0)), normalised=True)
        spatialFeHDist[:,i] = FeHdist(FeH)
        fH2Odist = SM2ISO(FeHdist, normalised=True)[0]
        spatialfH2ODist[:,i] = fH2Odist(fH2O)
        # if i == 85:
            # fig, ax = plt.subplots()
            # ax.plot(FeH, FeHdist(FeH))
            # fig, ax = plt.subplots()
            # ax.plot(fH2O, fH2Odist(FeH))
    
    fig, ax = plt.subplots()
    ax.plot(R, densityDist)
    ax.set_xlabel('R/kpc')
    ax.set_ylabel(r'Normalised [Fe/H] distribution ($\mathrm{dex}^{-1}$)')
    saveFig(fig, 'densityR')
    
    fig, ax = plt.subplots()
    ax.plot(R, densityDist*R)
    ax.set_xlabel('R/kpc')
    ax.set_ylabel('stellar mass density dist times R')
    saveFig(fig, 'densityTimesR')
    
    fig, ax = plt.subplots()
    ax.imshow(spatialFeHDist, origin='lower', extent=[R[0], R[-1], FeH[0], FeH[-1]], aspect='auto')
    ax.plot([R[0],R[-1]], [-0.4,-0.4], color ='C2')
    ax.plot([R[0],R[-1]], [ 0.4, 0.4], color ='C2')
    #ax.vlines(myUtils.R_Sun, ymin=FeH[0], ymax=FeH[-1], color='C0')
    ax.set_xlabel('R/kpc')
    ax.set_ylabel(r'Normalised [Fe/H] distribution ($\mathrm{dex}^{-1}$)')
    saveFig(fig, 'FeHR')
    
    fig, ax = plt.subplots()
    ax.imshow(spatialfH2ODist, origin='lower', extent=[R[0], R[-1], fH2O[0], fH2O[-1]], aspect='auto')
    ax.set_xlabel('R/kpc')
    ax.set_ylabel(r'Normalised ISO $f_\mathrm{H_2O}$ distribution')
    saveFig(fig, 'fH2OR')
    
    # fig,ax = plt.subplots()
    # ax.plot(StellarMassDM.histWithin(4))
    # ax.plot(StellarMassDM.integratedHist() - StellarMassDM.histWithin(4))
    


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
if SMOOTH_FeH:
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
            
else:
    """linear interpolation of BB20"""
    comp = lambda FeH: np.interp(FeH, xp=FeH_p, fp=fH2O_p)
    compInv = lambda fH2O: np.interp(fH2O, xp=np.flip(fH2O_p), fp=np.flip(FeH_p), left=np.nan, right=np.nan)
    compDeriv = lambda FeH: (comp(FeH+0.0001)-comp(FeH))/0.0001

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

if PROPZ:
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
else:
    def SM2ISO(FeHdist, alpha=1, normalised=False):
        normFactor = alpha*quad(FeHdist, -np.inf, np.inf, limit=200)[0] if normalised else 1
        ISOdist = lambda fH2O: -alpha*FeHdist(compInv(fH2O))/(normFactor*compDeriv(compInv(fH2O)))
        lowerEndCount = alpha*quad(FeHdist, FeHhigh, np.inf, limit=200)[0]/normFactor
        upperEndCount = alpha*quad(FeHdist, -np.inf, FeHlow, limit=200)[0]/normFactor
        return (ISOdist, lowerEndCount, upperEndCount)
    
# def bindex(edges, value):
#     if value.ndim==0:
#         return np.nonzero(edges <= value)[0][-1]
#     else:
#         return np.array([np.nonzero(edges <= v)[0][-1] for v in value])



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
