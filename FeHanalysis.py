import os
from functools import partial
import pickle

import numpy as np
import matplotlib.pyplot as plt

import myUtils

# for 1D bins - assumed ordered, non-overlapping, equal size

SMOOTH=True
FINE=True
POLYDEG=3

binsDir = '/Users/hopkinsm/data/APOGEE/bins'


def main():
    binList = myUtils.binsToUse
    
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
        NRG2mass[i] = loadNRG2mass(binDict)
        
    #NRGDM = densityModel(FeHEdges, np.exp(FeHnumberlogA)/FeHWidths, aR, az)
    StellarMassDM = densityModel(FeHedges, NRG2mass*np.exp(FeHnumberlogA)/FeHwidths, aR, az)
    FeHdist_Sun = StellarMassDM.distribution()
    FeHdist_GalCentre = StellarMassDM.distribution(position=(0,0))
    FeHdist_integrated = StellarMassDM.integrated()
    
    fH2Oedges = myUtils.arr((0.0, 0.6, (0.005 if FINE else 0.05)))
    fH2Omidpoints = (fH2Oedges[:-1] + fH2Oedges[1:])/2
    fH2Owidths = fH2Oedges[1:] - fH2Oedges[:-1]
    
    fH2Odist_Sun = getfH2Odist(fH2Oedges, FeHedges, FeHdist_Sun)
    fH2Odist_GalCentre = getfH2Odist(fH2Oedges, FeHedges, FeHdist_GalCentre)
    fH2Odist_integrated = getfH2Odist(fH2Oedges, FeHedges, FeHdist_integrated)
    
    #plotting
    FeH_plot = np.linspace(-1, 0.5, 15*5+1)
    
    fig, ax = plt.subplots()
    ax.plot(FeH_plot, comp(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,0.5], color=f'C{i}')
    ax.set_xlim([-1,0.5])
    ax.set_xlabel(r'$[Fe/H]$')
    
    fig, ax = plt.subplots()
    ax.plot(FeH_plot[:-1], np.diff(comp(FeH_plot))/np.diff(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,-1], color=f'C{i}')
    ax.set_xlim([-1,0.5])
    ax.set_xlabel(r'[Fe/H]')
    
    # fig, ax = plt.subplots()
    # ax.plot(fH2O_plot, compInv(fH2O_plot))
    # for i, FeH in enumerate(FeHEdges):
    #     ax.plot([comp(FeH),comp(FeH)], [0,0.1], color=f'C{i}') #colours correspond on multiple graphs
    
    #at Sun
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHdist_Sun, width = FeHwidths)
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,1e7], color=f'C{i}')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (dM/dVdFeH)')
    ax.set_title('Solar neighborhood')
    saveFig(fig, 'localFeH')
    
    fig, ax = plt.subplots()
    ax.bar(fH2Omidpoints, fH2Odist_Sun, width=fH2Owidths)
    for i, FeH in enumerate(FeHedges):
        ax.plot([comp(FeH),comp(FeH)], [0,1e8], color=f'C{i}') #colours correspond on multiple graphs
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (dN/dVdfH2O)')
    ax.set_title('Solar neighborhood')
    
    
    # At Gal Centre
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHdist_GalCentre, width = FeHwidths)
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,1e9], color=f'C{i}')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (dM/dVdFeH)')
    ax.set_title('Galactic Centre')
    
    fig, ax = plt.subplots()
    ax.bar(fH2Omidpoints, fH2Odist_GalCentre, width=fH2Owidths)
    for i, FeH in enumerate(FeHedges):
        ax.plot([comp(FeH),comp(FeH)], [0,1e9], color=f'C{i}') #colours correspond on multiple graphs
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (dN/dVdfH2O)')
    ax.set_title('Galactic Centre')
        
    # Integrated
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHdist_integrated, width = FeHwidths)
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,1e10], color=f'C{i}')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (dM/dFeH)')
    ax.set_title('Integrated')
    
    fig, ax = plt.subplots()
    ax.bar(fH2Omidpoints, fH2Odist_integrated, width=fH2Owidths)
    for i, FeH in enumerate(FeHedges):
        ax.plot([comp(FeH),comp(FeH)], [0,1e11], color=f'C{i}') #colours correspond on multiple graphs
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (dN/dfH2O)')
    ax.set_title('Integrated')
    
    
    # EAGLE
    EAGLE_data = np.loadtxt('/Users/hopkinsm/data/APOGEE/input_data/EAGLE_MW_L0025N0376_REFERENCE_ApogeeRun_30kpc_working.dat') 
    # List of star particles
    EAGLE_mass = EAGLE_data[:,9]
    EAGLE_FeH = EAGLE_data[:,14]
    EAGLEedges = myUtils.arr((-2.575, 1.525, 0.1))
    EAGLEmidpoints = (EAGLEedges[:-1] + EAGLEedges[1:])/2
    EAGLEwidths = EAGLEedges[1:] - EAGLEedges[:-1]
    FeHdist_EAGLE = np.histogram(EAGLE_FeH, bins=EAGLEedges, weights=EAGLE_mass/EAGLEwidths[0])[0]

    fig, ax = plt.subplots()
    ax.bar(EAGLEmidpoints, FeHdist_EAGLE, width=EAGLEwidths)
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('stellar mass distribution (dM/d[Fe/H])')
    ax.set_title('EAGLE')
    
    fH2Odist_EAGLE = getfH2Odist(fH2Oedges, EAGLEedges, FeHdist_EAGLE)
    
    fig, ax = plt.subplots()
    ax.bar(fH2Omidpoints, fH2Odist_EAGLE, width=fH2Owidths)
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (dN/dfH2O)')
    ax.set_title('EAGLE')
    
    #Comparisons
    #Integrated vs solar neighborhood
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHdist_integrated/sum(FeHwidths*FeHdist_integrated), width = FeHwidths, alpha=0.5, label='Integrated')
    ax.bar(FeHmidpoints, FeHdist_Sun/sum(FeHwidths*FeHdist_Sun), width = FeHwidths, alpha=0.5, label='Local')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (normalised)')
    ax.legend()
    
    fig, ax = plt.subplots()
    ax.bar(fH2Omidpoints, fH2Odist_integrated/sum(fH2Owidths*fH2Odist_integrated), width=fH2Owidths, alpha=0.5, label='Integrated')
    ax.bar(fH2Omidpoints, fH2Odist_Sun/sum(fH2Owidths*fH2Odist_Sun),        width=fH2Owidths, alpha=0.5, label='Local')
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (normalised)')
    ax.legend()
    
    
    #Integrated vs EAGLE
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHdist_integrated/sum(FeHwidths*FeHdist_integrated), width = FeHwidths, alpha=0.5, label='Integrated')
    ax.bar(EAGLEmidpoints, FeHdist_EAGLE/sum(EAGLEwidths*FeHdist_EAGLE), width = EAGLEwidths, alpha=0.5, label='EAGLE')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (normalised)')
    ax.legend()
    
    fig, ax = plt.subplots()
    ax.bar(fH2Omidpoints, fH2Odist_integrated/sum(fH2Owidths*fH2Odist_integrated), width=fH2Owidths, alpha=0.5, label='Integrated')
    ax.bar(fH2Omidpoints, fH2Odist_EAGLE/sum(fH2Owidths*fH2Odist_EAGLE),        width=fH2Owidths, alpha=0.5, label='EAGLE')
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (normalised)')
    ax.legend()

    


def saveFig(fig, name):#could put in git hash
    path = '/Users/hopkinsm/Documents/APOGEE/plots/'+name
    fig.savefig(path)


def comp(FeH):
    """Chosen smooth version as stops spurios bumps in ISO distribution
    Means extrema aren't extrema of fH2O_p, instead 0.063856, 0.513264 for order 2"""
    FeH_p = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
    
    if not SMOOTH:
        return np.interp(FeH, xp=FeH_p, fp=fH2O_p)
    else:
        p = np.polynomial.polynomial.Polynomial.fit(FeH_p, fH2O_p, POLYDEG)
        return np.where(FeH_p[0]<=FeH, np.where(FeH<FeH_p[-1], p(FeH), p(FeH_p[-1])), p(FeH_p[0]))

# def compInv(fH2O):
#     FeH_p = np.flip(np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]))
#     fH2O_p = np.flip(np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516]))
#     return np.interp(fH2O, xp=fH2O_p, fp=FeH_p, left=np.nan, right=np.nan)

def loadFitResults(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'fit_results.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

def loadNRG2mass(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'NRG2mass.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

def getfH2Odist(fH2OEdges, FeHEdges, FeHmassDist):
    alpha = 1 # unknown constant of proportionality between number of ISOs produces and stellar mass
    fH2Odist = np.zeros(len(fH2OEdges)-1)
    NpointsPerBin = int(50*(FeHEdges[1]-FeHEdges[0])/(fH2OEdges[1]-fH2OEdges[0]))
    
    fH2Owidth = fH2OEdges[1]-fH2OEdges[0]
    FeHarray = np.linspace(FeHEdges[0], FeHEdges[-1], NpointsPerBin*(len(FeHEdges)-1), endpoint=False)
    FeHarray += (FeHarray[1] - FeHarray[0])/2
    FeHarrayWidth = FeHarray[1]-FeHarray[0]
    # evenly spaces NpointsPerBin points in each bin offset from edges
    fH2Oarray = comp(FeHarray)
    
    for j in range(len(fH2Odist)):
        indexArray = np.nonzero((fH2OEdges[j] <= fH2Oarray)&(fH2Oarray < fH2OEdges[j+1]))[0]//NpointsPerBin
        # finds index of points in FeHarray, integer division changes them to index in FeHmassDistAmp bins
        fH2Odist[j] += (alpha*FeHarrayWidth/fH2Owidth) * FeHmassDist[indexArray].sum()
    return fH2Odist 
    
    
def bindex(edges, value):
    return np.nonzero(edges<=value)[0][-1]
        
    
    
    
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
        # print(len(self.edges),
        # len(self.widths),
        # len(self.midpoints),
        # len(self.distAmp),
        # len(self.aR),
        # len(self.az))

    def distribution(self, position=(myUtils.R_Sun, myUtils.z_Sun), normalised=False):
        R, z = position
        dist = self.distAmp*np.exp( - self.aR*(R-myUtils.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return dist
        else:
            return dist/sum(self.widths*self.distribution(position))
    
    def integrated(self, normalised=False):
        dist = 4*np.pi*self.distAmp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az)
        if not normalised:
            return dist
        else:
            return dist/sum(self.widths*self.integrated())
        

# def binParamSpaceVol(binDict):
#     size=1
#     for limits in binDict.values():
#         size*=(limits[1]-limits[0])
#     return size
    
    
if __name__=='__main__':
    main()
