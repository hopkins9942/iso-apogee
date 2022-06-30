# on mac, analysing data downloaded off cluster - if possible, avoid having to install apogee package stuff
# made onMac venv for this

import pickle
from math import isclose
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt


binsDir = '/Users/hopkinsm/FromARC20220628/bins/'
outputDir = '/Users/hopkinsm/FromARC20220628/onMacOutput/'


GC_frame = coord.Galactocentric() #adjust parameters here if needed
z_Sun = GC_frame.z_sun.to(u.kpc).value # .value removes unit, which causes problems with pytorch
R_Sun = np.sqrt(GC_frame.galcen_distance.to(u.kpc).value**2 - z_Sun**2)

def arr(gridParams):
    start, stop, step = gridParams
    arr = np.arange(round((stop-start)/step)+1)*step+start
    assert isclose(arr[-1],stop) # will highlight both bugs and when stop-start is not multiple of step
    return arr

def binName(binDict):
    """
    binDict
    """
    binDict = dict(sorted(binDict.items()))
    # ensures same order independant of construction. I think sorts according to ASCII value of first character of key
    return '_'.join(['_'.join([key, f'{limits[0]:.3f}', f'{limits[1]:.3f}']) for key,limits in binDict.items()])
    #Note: python list comprehensions are cool

def loadFitResults(binDict):
    path = binsDir + binName(binDict) + '/fit_results.dat'
    with open(path, 'rb') as f:
        return pickle.load(f)

def loadNRG2mass(binDict):
    path = binsDir + binName(binDict) + '/NRG2mass.dat'
    with open(path, 'rb') as f:
        return pickle.load(f)

def binParamSpaceVol(binDict):
    size=1
    for limits in binDict.values():
        size*=(limits[1]-limits[0])
    return size


__FeH_edges = arr((-1.025, 0.475, 0.1))
binsToUse = [{'FeH': (__FeH_edges[i], __FeH_edges[i+1])} for i in range(len(__FeH_edges)-1)]

muMin = 4.0
muMax = 17.0
muStep = 0.1
muGridParams = (muMin, muMax, muStep)

def plotDistributions(bins, R=R_Sun, z=z_Sun, label='FeH'):
    Nbins = len(bins)
    results_array = np.zeros((Nbins,3))
    NRGperVolperParamSpaceVol_array = np.zeros(Nbins)
    NRG2mass_array = np.zeros(Nbins)
    totalMassperParamSpaceVol_array = np.zeros(Nbins)
    for i in range(Nbins):
        binDict = bins[i]
        results_array[i,:] = loadFitResults(binDict)
        logNuSun, aR, az = results_array[i,:]
        NRGperVolperParamSpaceVol = np.exp(logNuSun - aR*(R-R_Sun) - az*np.abs(z))/binParamSpaceVol(binDict)
        NRGperVolperParamSpaceVol_array[i] = NRGperVolperParamSpaceVol
        NRG2mass = loadNRG2mass(binDict)
        NRG2mass_array[i] = NRG2mass
        totalMassperParamSpaceVol_array[i] = 4*np.pi*np.exp(logNuSun + aR*R_Sun)*NRG2mass/(aR**2 * az * binParamSpaceVol(binDict))
    massperVolperParamSpaceVol_array = NRGperVolperParamSpaceVol_array*NRG2mass_array

    if all('FeH' in bins[i] for i in range(Nbins)): # may need [ ] 
        # Bins only in one variable, everything 1D
        key = 'FeH'
        
        savePath = outputDir
        midpoints = np.array([(bin[key][0] + bin[key][1])/2 for bin in bins])
        widths = np.array([(bin[key][1] - bin[key][1]) for bin in bins])
        edges = np.append((midpoints - widths/2), midpoints[-1]+widths[-1]/2)

        fig, ax = plt.subplots()
        ax.bar(midpoints, np.exp(results_array[:,0]), width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('exp(logNuSun)')
        fig.savefig(savePath+'expLogNuSun.png')

        fig, ax = plt.subplots()
        ax.bar(midpoints, 1/results_array[:,1], width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Scale length /kpc')
        fig.savefig(savePath+'h_R.png')

        fig, ax = plt.subplots()
        ax.bar(midpoints, 1/results_array[:,2], width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Scale height /kpc')
        fig.savefig(savePath+'h_z.png')

        fig, ax = plt.subplots()
        ax.bar(midpoints, NRGperVolperParamSpaceVol_array, width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Red giant number density distribution at Sun')
        fig.savefig(savePath+'NRGperVolperParamSpaceVol.png')

        fig, ax = plt.subplots()
        ax.bar(midpoints, massperVolperParamSpaceVol_array, width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Mass density distribution at Sun')
        fig.savefig(savePath+'massperVolperParamSpaceVol.png')
    
        fig, ax = plt.subplots()
        ax.bar(midpoints, totalMassperParamSpaceVol_array, width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Total mass distribution (dM/d[Fe/H])')
        ax.set_title('APOGEE')
        fig.savefig(savePath+'totalMassperParamSpaceVol.png')

        EAGLE_data = np.loadtxt('/Users/hopkinsm/FromARC20220628/input/EAGLE_MW_L0025N0376_REFERENCE_ApogeeRun_30kpc_working.dat') 
        EAGLE_mass = EAGLE_data[:,9]
        EAGLE_FeH = EAGLE_data[:,14]
        EAGLE_binWidth = 0.1
        EAGLE_bins = arr((-2.525, 1.475, EAGLE_binWidth))
        
        fig, ax = plt.subplots()
        EAGLE_totalMassperParamSpaceVol_array, EAGLE_binEdges, *_ = ax.hist(EAGLE_FeH, weights=EAGLE_mass/EAGLE_binWidth, bins=EAGLE_bins)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Total mass distribution (dM/d[Fe/H])')
        ax.set_title('EAGLE')
        fig.savefig(savePath+'EAGLE.png')    

        EAGLE_midpoints = (EAGLE_binEdges[1:]+ EAGLE_binEdges[:-1])/2
        fig, ax = plt.subplots()
        ax.plot(      midpoints,       totalMassperParamSpaceVol_array, label='APOGEE')
        ax.plot(EAGLE_midpoints, EAGLE_totalMassperParamSpaceVol_array, label='EAGLE')
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Total mass distribution (dM/d[Fe/H])')
        ax.legend()
        fig.savefig(savePath+'joint.png')

        
        #define function for mappping mass distribution to fH20
        def fH20_distribution(FeH_edges, massPerFeH):#, fH20_edges=arr(0,0.52,0.01)):
            fH20_width=0.01
            fH20_edges=arr(0,0.52,fH20_width)

            FeH_p = [-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4]
            fH2O_p = [ 0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516]
            comp = partial(np.interp, xp=FeH_p, fp=fH2O_p)
            FeH = np.linspace(FeH_edges[0], FeH_edges[-1], (FeH_edges[-1] - FeH_edges[0])/(fH20_width/10))
            fH20 = comp(FeH)

     ??? SORT ALL THIS OUT ????

        fig, ax = plt.subplots()
        ax.plot(FeH, fH2O)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Water fraction')
        fig.savefig(savePath + 'BB20.png')

        dfH20dFeH_array = np.diff(fH20)/0.1
        grad = lambda FeH: 
        rho_funct()

        # fH20:
        fH2OBinEdges = arr(0,0.52, )




    else:
        raise NotImplementedError


     

plotDistributions(binsToUse)

