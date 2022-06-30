# on mac, analysing data downloaded off cluster - if possible, avoid having to install apogee package stuff
# made onMac venv for this

import pickle
from math import isclose
from functools import partial
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
    #assert isclose(arr[-1],stop) # will highlight both bugs and when stop-start is not multiple of step
    arr = np.around(arr, 4) # need more exact boundaries
    assert arr[-1]==stop
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


class composition_model:
    def __init__(self):
        pass
    _FeH_p = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4])
    _fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
    _comp = partial(np.interp, xp=_FeH_p, fp=_fH2O_p)

    def __call__(self, **kwargs):
        return self._comp(kwargs['FeH'])

    def inverse(self):
        pass

    def gradient(self, **kwargs):
        pass
        

fH2O_funct = composition_model()


def convert(massModel, compositionModel, fH2O_params=(0,0.52,0.05)):
    """ works by covereing binned parameter space in points, and calculating contribtuion of each to corresponding fH2O bin"""
    alpha = 1
    fH2O_binEdges = arr(fH2O_params)
    fH2O_step = fH2O_binEdges[1] - fH2O_binEdges[0]
    fH2O_binValues = np.zeros(len(fH2O_binEdges))
    NPointsPerCoordPerBin = 10 # for FeH bins of width 0.1 and fH2O widhs of 0.05, 10 gives 5 FeH points per fH2O bin

    for bin in massModel.bins:
        Nkeys = len(bin.keys())
        key_list = list(bin.keys())
        paramSpaceVolElement = binParamSpaceVol(bin)/(NPointsPerCoordPerBin*Nkeys)
        coordinateArrays = [np.linspace(limits[0], limits[1], NPointsPerCoordPerBin, endpoint=False)
                            for limits in bin.values()]
        # coordinateArrays[i] is array spanning ith coordinate range of bin
        for i in range(Nkeys*NPointsPerCoordPerBin):
            indices = np.unravel_index(i, shape=[NPointsPerCoordPerBin]*Nkeys)
            point = {key_list[j]: coordinateArrays[j][indices[j]] for j in range(Nkeys)}

            fH2O_index = np.where(compositionModel(**point) >= fH2O_binEdges)[0].max()
            fH2O_binValues += alpha * massModel(**point) * paramSpaceVolElement/fH2O_step
    return fH2O_binEdges, fH2O_binValues
        
class distributionModel:
    def __init__(self, bins, values):
        self.bins = bins
        self.values = values
    def __call__(self, **kwargs):
        index = binIndex(self.bins, **kwargs)
        return self.values[index]

class doubleExpDensityModel(distributionModel):
    def __init__(self, bins, parameters, evaluate = (R_Sun, z_Sun)):
        self.bins = bins
        self.logAmp = parameters[:,0]
        self.aR = parameters[:,1]
        self.az = parameters[:,2]
        
        if len(evaluate)==2:
            R, z = evaluate
            values = np.exp(self.logAmp - self.aR*(R-R_Sun) - self.az*np.abs(z))
        
        elif evaluate=='integrated':
            values = 4*np.pi*np.exp(self.logAmp + self.aR*R_Sun)/(self.aR**2 * self.az)
        else:
            raise ValueError("input valid value")
        super().__init__(bins,values)


def binIndex(bins, **kwargs):
    """
    takes kwargs of values, finds bin those values lie in
    Assumes no overlapping bins
    """
    for i, bin in enumerate(bins):
        isIn = True
        for key, limits in bin.items():
            isIn &= (limits[0]<=kwargs[key]) & (kwargs[key]<limits[1])
        if isIn: return i
    return np.nan # not in any bin 

__FeH_edges = arr((-1.025, 0.475, 0.1))
binsToUse = [{'FeH': (__FeH_edges[i], __FeH_edges[i+1])} for i in range(len(__FeH_edges)-1)]



muMin = 4.0
muMax = 17.0
muStep = 0.1
muGridParams = (muMin, muMax, muStep)

def plotDistributions(bins, R=R_Sun, z=z_Sun, label='FeH'):
    Nbins = len(bins)
    results_array = np.zeros((Nbins,3))
    #NRGperVolperParamSpaceVol_array = np.zeros(Nbins)
    NRG2mass_array = np.zeros(Nbins)
    #totalMassperParamSpaceVol_array = np.zeros(Nbins)
    for i in range(Nbins):
        binDict = bins[i]
        results_array[i,:] = loadFitResults(binDict)
        #logNuSun, aR, az = results_array[i,:]
        #NRGperVolperParamSpaceVol = np.exp(logNuSun - aR*(R-R_Sun) - az*np.abs(z))/binParamSpaceVol(binDict)
        #NRGperVolperParamSpaceVol_array[i] = NRGperVolperParamSpaceVol
        NRG2mass_array[i] = loadNRG2mass(binDict)
        #totalMassperParamSpaceVol_array[i] = 4*np.pi*np.exp(logNuSun + aR*R_Sun)*NRG2mass/(aR**2 * az * binParamSpaceVol(binDict))
    #massperVolperParamSpaceVol_array = NRGperVolperParamSpaceVol_array*NRG2mass_array

    binSize_array = np.array([binParamSpaceVol(bin) for bin in bins])
    logNuSun_modifier = np.zeros_like(results_array)
    logNuSun_modifier[:,0] = np.log(1/binSize_array)
    NRG_density = doubleExpDensityModel(bins, results_array + logNuSun_modifier)
    logNuSun_modifier[:,0] = np.log(NRG2mass_array/binSize_array)
    local_mass_density = doubleExpDensityModel(bins, results_array + logNuSun_modifier)

    integrated_mass = doubleExpDensityModel(bins, results_array + logNuSun_modifier, evaluate='integrated')

    if all('FeH' in bins[i] for i in range(Nbins)): # may need [ ] 
        # Bins only in one variable, everything 1D
        key = 'FeH'
        
        savePath = outputDir
        midpoints_array = np.array([(bin[key][0] + bin[key][1])/2 for bin in bins])
        midpoints_dicts = [{key:midpoint} for midpoint in midpoints_array]
        widths = np.array([(bin[key][1] - bin[key][1]) for bin in bins])
        edges = np.append((midpoints_array - widths/2), midpoints_array[-1]+widths[-1]/2)

        fig, ax = plt.subplots()
        ax.bar(midpoints_array, np.exp(results_array[:,0]), width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('exp(logNuSun)')
        fig.savefig(savePath+'expLogNuSun.png')

        fig, ax = plt.subplots()
        ax.bar(midpoints_array, 1/results_array[:,1], width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Scale length /kpc')
        fig.savefig(savePath+'h_R.png')

        fig, ax = plt.subplots()
        ax.bar(midpoints_array, 1/results_array[:,2], width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Scale height /kpc')
        fig.savefig(savePath+'h_z.png')

        fig, ax = plt.subplots()
        ax.bar(midpoints_array, [NRG_density(**midpoints_dicts[i]) for i in range(Nbins)], width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Red giant number density distribution at Sun')
        fig.savefig(savePath+'NRGperVolperParamSpaceVol.png')

        fig, ax = plt.subplots()
        ax.bar(midpoints_array, [local_mass_density(**midpoints_dicts[i]) for i in range(Nbins)], width=widths)
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Mass density distribution at Sun')
        fig.savefig(savePath+'massperVolperParamSpaceVol.png')
    
        fig, ax = plt.subplots()
        ax.bar(midpoints_array, [integrated_mass(**midpoints_dicts[i]) for i in range(Nbins)], width=widths)
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
        ax.plot(midpoints_array, [integrated_mass(**midpoints_dicts[i]) for i in range(Nbins)], label='APOGEE')
        ax.plot(EAGLE_midpoints, EAGLE_totalMassperParamSpaceVol_array, label='EAGLE')
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('Total mass distribution (dM/d[Fe/H])')
        ax.legend()
        fig.savefig(savePath+'joint.png')

        
        #define function for mappping mass distribution to fH20
        #def fH20_distribution(FeH_edges, massPerFeH):#, fH20_edges=arr(0,0.52,0.01)):
        #    fH20_width=0.01
        #    fH20_edges=arr(0,0.52,fH20_width)


     #?? SORT ALL THIS OUT ????

        #fig, ax = plt.subplots()
        #ax.plot(FeH, fH2O)
        #ax.set_xlabel('[Fe/H]')
        #ax.set_ylabel('Water fraction')
        #fig.savefig(savePath + 'BB20.png')

        #dfH20dFeH_array = np.diff(fH20)/0.1
        #grad = lambda FeH: 
        #rho_funct()

        # fH20:
        #fH2OBinEdges = arr(0,0.52, )




    else:
        raise NotImplementedError



     

plotDistributions(binsToUse)

