import os
from functools import partial
import pickle

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt

import myUtils

# for 1D bins - assumed ordered, non-overlapping, equal size

SMOOTH_FeH=True
#FINE=True
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
    
    FeHhist_Sun = StellarMassDM.hist()
    FeHhist_GalCentre = StellarMassDM.hist(position=(0,0))
    FeHhist_integrated = StellarMassDM.integratedHist()
    
    # fH2Oedges = myUtils.arr((0.0, 0.6, (0.005 if FINE else 0.05)))
    # fH2Omidpoints = (fH2Oedges[:-1] + fH2Oedges[1:])/2
    # fH2Owidths = fH2Oedges[1:] - fH2Oedges[:-1]
    
    # fH2Odist_Sun = getfH2Ohist(fH2Oedges, FeHedges, FeHdist_Sun)
    # fH2Odist_GalCentre = getfH2Ohist(fH2Oedges, FeHedges, FeHdist_GalCentre)
    # fH2Odist_integrated = getfH2Ohist(fH2Oedges, FeHedges, FeHdist_integrated)
    
    # smoothfH2Odist_Sun = getSmoothfH2Ohist(fH2Oedges, FeHedges, FeHdist_Sun)
    # smoothfH2Odist_GalCentre = getSmoothfH2Ohist(fH2Oedges, FeHedges, FeHdist_GalCentre)
    # smoothfH2Odist_integrated = getSmoothfH2Ohist(fH2Oedges, FeHedges, FeHdist_integrated)
    
    fH2Odist_Sun = SM2ISOdist(hist2callable(FeHedges, FeHhist_Sun))
    fH2Odist_GalCentre = SM2ISOdist(hist2callable(FeHedges, FeHhist_GalCentre))
    fH2Odist_integrated = SM2ISOdist(hist2callable(FeHedges, FeHhist_integrated))
    
    smoothfH2Odist_Sun = SM2ISOdist(smoothDist(FeHedges, FeHhist_Sun))
    smoothfH2Odist_GalCentre = SM2ISOdist(smoothDist(FeHedges, FeHhist_GalCentre))
    smoothfH2Odist_integrated = SM2ISOdist(smoothDist(FeHedges, FeHhist_integrated))
    
    #smoothMassSun = smoothDist(FeHedges, FeHdist_Sun)
    
    
    
    
    #plotting
    FeH_plot = np.linspace(-1, 0.5, 15*20+1)
    fH2O_plot = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
    
    
    fig, ax = plt.subplots()
    ax.plot(FeH_plot, comp(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,0.5], color=f'C{i}')
    ax.set_xlim([-1,0.5])
    ax.set_xlabel(r'[Fe/H]')
    saveFig(fig,'comp.png')
    
    fig, ax = plt.subplots()
    ax.plot(FeH_plot, compDeriv(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,-1], color=f'C{i}')
    ax.set_xlim([-1,0.5])
    ax.set_xlabel(r'[Fe/H]')
    saveFig(fig,'compDeriv.png')
    
    # fig, ax = plt.subplots()
    # ax.plot(fH2O_plot, compInv(fH2O_plot))
    # for i, FeH in enumerate(FeHEdges):
    #     ax.plot([comp(FeH),comp(FeH)], [0,0.1], color=f'C{i}') #colours correspond on multiple graphs
    
    #at Sun
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHhist_Sun, width = FeHwidths)
    ax.plot(FeH_plot, smoothDist(FeHedges, FeHhist_Sun)(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,1e7], color=f'C{i}')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (dM/dVdFeH)')
    ax.set_title('Solar neighborhood')
    saveFig(fig,'localMass.png')

    
    fig,ax = plt.subplots()
    ax.bar(FeHmidpoints, [quad(smoothDist(FeHedges, FeHhist_Sun), FeHedges[i], FeHedges[i+1])[0]/FeHwidths[i]
                          for i in range(len(FeHmidpoints))], width = FeHwidths)
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,1e7], color=f'C{i}')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (dM/dVdFeH)')
    ax.set_title('Solar neighborhood')
    saveFig(fig,'checkLocalMass.png')
    print(np.array([quad(smoothDist(FeHedges, FeHhist_Sun), FeHedges[i], FeHedges[i+1])[0]/FeHwidths[i]
                          for i in range(len(FeHmidpoints))]) - FeHhist_Sun)
    
    fig, ax = plt.subplots()
    ax.plot(fH2O_plot, fH2Odist_Sun(fH2O_plot))
    ax.plot(fH2O_plot, smoothfH2Odist_Sun(fH2O_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([comp(FeH),comp(FeH)], [0,1e8], color=f'C{i}') #colours correspond on multiple graphs
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (dN/dVdfH2O)')
    ax.set_title('Solar neighborhood')
    saveFig(fig,'localISOs.png')
    
    
    # At Gal Centre
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHhist_GalCentre, width = FeHwidths)
    ax.plot(FeH_plot, smoothDist(FeHedges, FeHhist_GalCentre)(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,1e9], color=f'C{i}')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (dM/dVdFeH)')
    ax.set_title('Galactic Centre')
    
    fig, ax = plt.subplots()
    ax.plot(fH2O_plot, fH2Odist_GalCentre(fH2O_plot))
    ax.plot(fH2O_plot, smoothfH2Odist_GalCentre(fH2O_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([comp(FeH),comp(FeH)], [0,1e9], color=f'C{i}') #colours correspond on multiple graphs
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (dN/dVdfH2O)')
    ax.set_title('Galactic Centre')
    saveFig(fig,'galCentreISOs.png')
    
        
    # Integrated
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHhist_integrated, width = FeHwidths)
    ax.plot(FeH_plot, smoothDist(FeHedges, FeHhist_integrated)(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,1e10], color=f'C{i}')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (dM/dFeH)')
    ax.set_title('Integrated')
    saveFig(fig,'integratedMass.png')
    
    fig, ax = plt.subplots()
    ax.plot(fH2O_plot, fH2Odist_integrated(fH2O_plot))
    ax.plot(fH2O_plot, smoothfH2Odist_integrated(fH2O_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([comp(FeH),comp(FeH)], [0,1e11], color=f'C{i}') #colours correspond on multiple graphs
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (dN/dfH2O)')
    ax.set_title('Integrated')
    saveFig(fig,'integratedISOs.png')
    
    
    # EAGLE
    EAGLE_data = np.loadtxt('/Users/hopkinsm/data/APOGEE/input_data/EAGLE_MW_L0025N0376_REFERENCE_ApogeeRun_30kpc_working.dat') 
    # List of star particles
    EAGLE_mass = EAGLE_data[:,9]
    EAGLE_FeH = EAGLE_data[:,14]
    EAGLEedges = myUtils.arr((-2.575, 1.525, 0.1))
    EAGLEmidpoints = (EAGLEedges[:-1] + EAGLEedges[1:])/2
    EAGLEwidths = EAGLEedges[1:] - EAGLEedges[:-1]
    FeHhist_EAGLE = np.histogram(EAGLE_FeH, bins=EAGLEedges, weights=EAGLE_mass/EAGLEwidths[0])[0]

    fig, ax = plt.subplots()
    ax.bar(EAGLEmidpoints, FeHhist_EAGLE, width=EAGLEwidths)
    ax.plot(FeH_plot, smoothDist(EAGLEedges, FeHhist_EAGLE)(FeH_plot))
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('stellar mass distribution (dM/d[Fe/H])')
    ax.set_title('EAGLE')
    saveFig(fig,'eagleMass.png')
    
    # fH2Odist_EAGLE = getfH2Ohist(fH2Oedges, EAGLEedges, FeHdist_EAGLE)
    fH2Odist_EAGLE = SM2ISOdist(hist2callable(EAGLEedges, FeHhist_EAGLE))
    smoothfH2Odist_EAGLE = SM2ISOdist(smoothDist(EAGLEedges, FeHhist_EAGLE))

    
    fig, ax = plt.subplots()
    ax.plot(fH2O_plot, fH2Odist_EAGLE(fH2O_plot))
    ax.plot(fH2O_plot, smoothfH2Odist_EAGLE(fH2O_plot))
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (dN/dfH2O)')
    ax.set_title('EAGLE')
    saveFig(fig,'eagleISOs.png')
    
    #Comparisons
    #Integrated vs solar neighborhood
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHdist_integrated/sum(FeHwidths*FeHdist_integrated), width = FeHwidths, alpha=0.5, label='Integrated')
    ax.bar(FeHmidpoints, FeHdist_Sun/sum(FeHwidths*FeHdist_Sun), width = FeHwidths, alpha=0.5, label='Local')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (normalised)')
    ax.legend()
    saveFig(fig,'localIntegratedMass.png')
    
    fig, ax = plt.subplots()
    ax.bar(fH2Omidpoints, fH2Odist_integrated/sum(fH2Owidths*fH2Odist_integrated), width=fH2Owidths, alpha=0.5, label='Integrated')
    ax.bar(fH2Omidpoints, fH2Odist_Sun/sum(fH2Owidths*fH2Odist_Sun),        width=fH2Owidths, alpha=0.5, label='Local')
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (normalised)')
    ax.legend()
    saveFig(fig,'localIntegratedISOs.png')
    
    
    #Integrated vs EAGLE
    fig, ax = plt.subplots()
    ax.bar(FeHmidpoints, FeHdist_integrated/sum(FeHwidths*FeHdist_integrated), width = FeHwidths, alpha=0.5, label='Integrated')
    ax.bar(EAGLEmidpoints, FeHdist_EAGLE/sum(EAGLEwidths*FeHdist_EAGLE), width = EAGLEwidths, alpha=0.5, label='EAGLE')
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel('Stellar mass distribution (normalised)')
    ax.legend()
    saveFig(fig,'eagleIntegratedMass.png')
    
    fig, ax = plt.subplots()
    ax.bar(fH2Omidpoints, fH2Odist_integrated/sum(fH2Owidths*fH2Odist_integrated), width=fH2Owidths, alpha=0.5, label='Integrated')
    ax.bar(fH2Omidpoints, fH2Odist_EAGLE/sum(fH2Owidths*fH2Odist_EAGLE),        width=fH2Owidths, alpha=0.5, label='EAGLE')
    ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    ax.set_ylabel('ISO distribution (normalised)')
    ax.legend()
    saveFig(fig,'eagleIntegratedISOs.png')

    


def saveFig(fig, name):#could put in git hash
    saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/testing/' #{SMOOTH}{FINE}{POLYDEG}/'
    os.makedirs(saveDir, exist_ok=True)
    path = saveDir+name
    fig.savefig(path)

# Defining comp:
FeH_p = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
compPoly = np.polynomial.polynomial.Polynomial.fit(FeH_p, fH2O_p, POLYDEG)
FeHlow = FeH_p[0]
FeHhigh = FeH_p[-1]
fH2Olow = compPoly(FeH_p[-1])
fH2Ohigh = compPoly(FeH_p[0])
if SMOOTH_FeH:
    comp = lambda FeH: np.where(FeHlow<=FeH, np.where(FeH<FeHhigh,
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
        # if FeHlow<=FeH<FeHhigh:
        #     return compPoly.deriv()(FeH)
        # else:
        #     return 0
        
            
else:
    comp = lambda FeH: np.interp(FeH, xp=FeH_p, fp=fH2O_p)
    compInv = lambda fH2O: np.interp(fH2O, xp=np.flip(fH2O_p), fp=np.flip(FeH_p), left=np.nan, right=np.nan)
    compDeriv = lambda FeH: (comp(FeH+0.0001)-comp(FeH))/0.0001

for x in [-0.4+0.0001, -0.2, 0, 0.2, 0.4-0.0001]:
    assert np.isclose(compInv(compPoly(x)), x) #checks inverse works


def loadFitResults(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'fit_results.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

def loadNRG2mass(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'NRG2mass.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

# def getfH2Ohist(fH2Oedges, FeHedges, FeHmassHist):
#     alpha = 1 # unknown constant of proportionality between number of ISOs produces and stellar mass
#     fH2Ohist = np.zeros(len(fH2Oedges)-1)
#     fH2Owidth = fH2Oedges[1]-fH2Oedges[0]
#     FeHwidth = FeHedges[1]-FeHedges[0]
#     NpointsPerBin = int(50*(FeHwidth)/(fH2Owidth))
#     # ensures sufficient points in each fH2O bin
#     # (= NpointsPerBin*fH2)width/FeHwidth)
    
#     FeHarray = np.linspace(FeHedges[0], FeHedges[-1], NpointsPerBin*(len(FeHedges)-1), endpoint=False)
#     FeHarray += (FeHarray[1] - FeHarray[0])/2
#     FeHarrayWidth = FeHarray[1]-FeHarray[0]
#     # evenly spaces NpointsPerBin points in each bin offset from edges
#     # note for smooth dist offset not necessary, but probably best kept
#     fH2Oarray = comp(FeHarray)
    
#     for j in range(len(fH2Ohist)):
#         indexArray = np.nonzero((fH2Oedges[j] <= fH2Oarray)&(fH2Oarray < fH2Oedges[j+1]))[0]//NpointsPerBin
#         # finds index of points in FeHarray, integer division changes them to index in FeHmassDistAmp bins
#         fH2Ohist[j] += (alpha*FeHarrayWidth/fH2Owidth) * FeHmassHist[indexArray].sum()
#     return fH2Ohist 

# def getSmoothfH2Ohist(fH2Oedges, FeHedges, FeHmassHist):
#     FeHmassDist = smoothDist(FeHedges, FeHmassHist)
    
#     alpha = 1 # unknown constant of proportionality between number of ISOs produces and stellar mass
#     fH2Ohist = np.zeros(len(fH2Oedges)-1)
#     fH2Owidth = fH2Oedges[1]-fH2Oedges[0]
#     FeHwidth = FeHedges[1]-FeHedges[0]
#     NpointsPerBin = int(50*(FeHwidth)/(fH2Owidth))
#     # ensures sufficient points in each fH2O bin
#     # (= NpointsPerBin*fH2)width/FeHwidth)
    
#     FeHarray = np.linspace(FeHedges[0], FeHedges[-1], NpointsPerBin*(len(FeHedges)-1), endpoint=False)
#     FeHarray += (FeHarray[1] - FeHarray[0])/2
#     FeHarrayWidth = FeHarray[1]-FeHarray[0]
#     # evenly spaces NpointsPerBin points in each bin offset from edges
#     # note for smooth dist offset not necessary, but probably best kept
#     fH2Oarray = comp(FeHarray)
    
#     for j in range(len(fH2Ohist)):
#         indexArray = np.nonzero((fH2Oedges[j] <= fH2Oarray)&(fH2Oarray < fH2Oedges[j+1]))[0]
#         # finds index of points in FeHarray
#         fH2Ohist[j] += (alpha*FeHarrayWidth/fH2Owidth) * FeHmassDist(FeHarray[indexArray]).sum()
#     return fH2Ohist 
    
def SM2ISOdist(FeHdist):
    alpha = 1
    func = lambda fH2O: -alpha*FeHdist(compInv(fH2O))/compDeriv(compInv(fH2O))
    return func
    
def bindex(edges, value):
    if value.ndim==0:
        return np.nonzero(edges <= value)[0][-1]
    else:
        return np.array([np.nonzero(edges <= v)[0][-1] for v in value])
        


def smoothDist(edges, histDist):
    widths = edges[1:] - edges[:-1]
    y = np.append(0, np.cumsum(widths*histDist))
    return CubicSpline(edges, y, bc_type='clamped', extrapolate=True).derivative()

    
def hist2callable(edges, hist):
    return (lambda value: hist[bindex(edges, value)])
    
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
        

# def binParamSpaceVol(binDict):
#     size=1
#     for limits in binDict.values():
#         size*=(limits[1]-limits[0])
#     return size
    
    
if __name__=='__main__':
    main()
