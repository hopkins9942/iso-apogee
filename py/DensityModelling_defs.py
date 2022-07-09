#from numbers import Number
import pickle
import os
#import datetime
import warnings
from math import isclose
from functools import partial

assert os.getenv('RESULTS_VERS')=='l33' # apogee should automatically use dr16 as long as this is correct

import apogee.select as apsel
import apogee.tools.read as apread
#import isochrones as iso
#import mwdust
#import tqdm

import numpy as np
import torch
#import pyro
#import pyro.distributions as distributions
import pyro.distributions.constraints as constraints
from pyro.distributions.util import broadcast_all
from pyro.distributions.torch_distribution import TorchDistribution
#from pyro.infer import SVI, Trace_ELBO
#from pyro.optim import Adam

import astropy.coordinates as coord
import astropy.units as u
#import corner
import matplotlib.pyplot as plt

import isochrones

_DEGTORAD = torch.pi/180
_ROOTDIR = "/home/sjoh4701/APOGEE/iso-apogee/"
_DATADIR = "/data/phys-galactic-isos/sjoh4701/APOGEE/"

GC_frame = coord.Galactocentric() #adjust parameters here if needed
z_Sun = GC_frame.z_sun.to(u.kpc).value # .value removes unit, which causes problems with pytorch
R_Sun = np.sqrt(GC_frame.galcen_distance.to(u.kpc).value**2 - z_Sun**2)


def makeBinDirs(): # a little faulty - when called by multiple scripts in parallel race condition occurs with if statement - could make one bin version for scripts which work on one bin each in parallel
    for binDict in binsToUse:
        path = _DATADIR+'bins/'+binName(binDict)
        if not os.path.exists(path):
            os.mkdir(path)

def arr(gridParams):
    start, stop, step = gridParams
    arr = np.arange(round((stop-start)/step)+1)*step+start
    assert isclose(arr[-1],stop) # will highlight both bugs and when stop-start is not multiple of diff
    return arr

def binName(binDict):
    """
    binDict
    """
    binDict = dict(sorted(binDict.items()))
    # ensures same order independant of construction. I think sorts according to ASCII value of first character of key
    return '_'.join(['_'.join([key, f'{limits[0]:.3f}', f'{limits[1]:.3f}']) for key,limits in binDict.items()])
    #Note: python list comprehensions are cool

__FeH_edges = arr((-1.575, 0.625, 0.1)) #0.625-09.725 has no APOGEE statsample stars, -1.575--1.475 has about 130
#__age_edges = np.array([-999.0,4.5,9.0,999.0]) # remember ages only good for FeH>-0.5
#__FeH_edges_for_age = arr((-0.475, 0.625, 0.1))
binsToUse = [{'FeH': (__FeH_edges[i], __FeH_edges[i+1])} for i in range(len(__FeH_edges)-1)]
#binsToUse = [{'FeH': (__FeH_edges_for_age[i], __FeH_edges_for_age[i+1]), 'age': (__age_edges[j], __age_edges[j+1])}
#    for i in range(len(__FeH_edges_for_age)-1) for j in range(len(__age_edges)-1)]

muMin = 4.0
muMax = 17.0
muStep = 0.1
muGridParams = (muMin, muMax, muStep)

#FeHBinEdges_array = [[-1.0,-0.75], [-0.75,-0.5], [-0.5,-0.25], [-0.25,0.0], [0.0,0.25], [0.25,0.5]]
#Nbins = len(FeHBinEdges_array)

#Nstars = 160205
#unadjusted_data = [[2.35200000e+03, 7.34630784e+00, 1.62761062e+00],
# [1.45120000e+04, 9.24297959e+00, 1.01059230e+00],
# [4.31350000e+04, 9.50945083e+00, 5.80609531e-01],
# [5.48160000e+04, 8.87167417e+00, 3.69494077e-01],
# [3.59720000e+04, 8.06852236e+00, 3.07138239e-01],
# [6.55000000e+03, 6.88423645e+00, 3.26783708e-01]]
#Nbad = 787
#unadjusted_results = [[6.710781574249268, 0.33299723267555237, 0.8103033900260925],
# [8.96414852142334, 0.26698002219200134, 1.2854876518249512],
# [10.64326000213623, 0.2955845296382904, 2.1618664264678955],
# [11.43671703338623, 0.38112685084342957, 3.333465099334717],
# [11.177633285522461, 0.45318296551704407, 4.000054359436035],
# [9.412787437438965, 0.5277802348136902, 4.067354202270508]]

# = [202.16782012, 204.26083429, 208.70700329, 216.78030342, 232.05002116, 243.72646311]

#mask_array = [((isogrid['logg'] > 1) & (isogrid['logg'] < 3)
#              & (isogrid['MH'] >  FeHBinEdges[0])
#              & (isogrid['MH'] <= FeHBinEdges[1]))
#              for FeHBinEdges in FeHBinEdges_array]

def calculateData(bins):
    """
    Calculates data for fit
    """
    allStar, statIndx = get_allStar_statIndx()

    print(len(allStar))
    print("Stat sample size: ", np.count_nonzero(statIndx))

    statSample = allStar[statIndx]
    mu, D, R, modz, gLon, gLat, x, y, z, FeH, MgFe, age = extract(statSample)
    in_mu_range = np.logical_and.reduce([(muMin<=mu), (mu<muMax)])
    #could be swapped in for calc_allStarSample_mask(dict(),statSample)
    Nstars = np.count_nonzero(in_mu_range)
    bad_indices = np.logical_and.reduce([(FeH<-9999), (MgFe<-9999), in_mu_range])
    # jth value is True, when star should be in my sample, but FeH or MgFe measurement failed
    # CHECK bad FeH and bad MgFe correlate, as if only selecting on FeH, could still use bad MgFe stars
    Nbad = np.count_nonzero(bad_indices)
    print("Number of bad stars in mu range: ", Nbad)
    print("Num of stars in mu range: ", Nstars)
    adjustment_factor = 1/(1-(Nbad/Nstars))
    print(adjustment_factor)
    adjusted_data = np.zeros([len(bins), 3])
    for i in range(len(bins)):
        in_bin = calc_allStarSample_mask(bins[i], statSample)
        # jth value is True when jth star lies in mu range and ith bin
        adjusted_data[i,0] = np.count_nonzero(in_bin)*adjustment_factor
        adjusted_data[i,1] = R[in_bin].mean()
        adjusted_data[i,2] = modz[in_bin].mean() #double check this
        with open(_DATADIR + 'bins/' + binName(bins[i]) + '/data.dat', 'wb') as f:
            pickle.dump(adjusted_data[i,:], f)
    print("Adjusted data: ", adjusted_data)

    return adjusted_data

def NRG2mass(binDict): # change to a get_ function which either loads or pickles
    isogrid = isochrones.newgrid()
    whole_bin_mask = calc_isogrid_mask(binDict, isogrid, RG_only=False)
    RG_bin_mask = calc_isogrid_mask(binDict, isogrid, RG_only=True)
    meanMass = np.average(isogrid[whole_bin_mask]['Mass'], weights=isogrid[whole_bin_mask]['weights'])
    RGfraction = isogrid[RG_bin_mask]['weights'].sum()/isogrid[whole_bin_mask]['weights'].sum()
    return meanMass/RGfraction

def binSize(binDict):
    """make this better"""
    size=1
    for limits in binDict.values():
        size*=(limits[1]-limits[0])
    return size

def postProcessing(bins):
    savePath = _DATADIR + 'outputs/DM_results/'
    #FeHBinEdges_array = np.array(FeHBinEdges_array)
    #isogrid = isochrones.newgrid()
    Nbins = len(bins)
    fitResults = np.zeros((Nbins,3))
    for i in range(Nbins):
        path = _DATADIR + 'bins/' + binName(bins[i]) + '/'
        with open(path + 'fit_results.dat', 'rb') as f:
            fitResults[i,:] = pickle.load(f)
    print(fitResults)

#    NRG2mass_array = np.zeros(Nbins)
#    for i in range(Nbins):
#        FeHBinEdges = FeHBinEdges_array[i]
#        binMask = ((FeHBinEdges[0] <= isogrid['MH'])
#                  &(isogrid['MH'] < FeHBinEdges[1]))
#        RGbinMask = binMask & (1 <= isogrid['logg']) & (isogrid['logg'] < 3)
#        meanMass = np.average(isogrid[binMask]['Mass'], weights=isogrid[binMask]['weights']) # Check with T$    print(meanMass)
#        RGfraction = isogrid[RGbinMask]['weights'].sum()/isogrid[binMask]['weights'].sum()
#        print(RGfraction)
#        NRG2mass_array[i] = meanMass/RGfraction
#    print(NRG2mass_array)

    NRG2mass_array = np.array([get_NRG2mass(bins[i]) for i in range(Nbins)])
    #binSize_array = np.array([binSize(bins[i]) for i in range(bins)]) # use this?
    left_edges = np.array([bins[i]['FeH'][0] for i in range(Nbins)])
    FeHBinEdges_array = np.zeros((Nbins,2))
    FeHBinEdges_array[:,0] = left_edges
    FeHBinEdges_array[:,1] = left_edges+(left_edges[1]-left_edges[0])
    #improve this

    massDensity = np.exp(fitResults[:,0])*NRG2mass_array*np.exp(-fitResults[:,2]*np.abs(z_Sun))/(FeHBinEdges_array[0,1]-FeHBinEdges_array[0,0])
    #mass density of all stars at solar position - per volume and per mettalicity
    hR = 1/fitResults[:,1]
    hz = 1/fitResults[:,2]
    print(massDensity)
    # CHeck by checking integrated mass density matches 10^10 galaxy mass
    #left_edges = FeHBinEdges_array[:,0]
    width = FeHBinEdges_array[0,1]-FeHBinEdges_array[0,0]

    fig, ax = plt.subplots()
    ax.bar(left_edges,massDensity,width=width, align='edge')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel(r'Mass density of stars around Sun /$\mathrm{M}_\odot \mathrm{kpc}^{-3}$')
    fig.savefig(savePath + 'rhoSun.png')

    fig, ax = plt.subplots()
    ax.bar(left_edges,hR,width=width, align='edge')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('Scale length/kpc')
    fig.savefig(savePath + 'hR.png')

    fig, ax = plt.subplots()
    ax.bar(left_edges,hz,width=width, align='edge')
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('Scale hight/kpc')
    fig.savefig(savePath + 'hz.png')

    #calc composition here - put as map in seperate function
    FeH_p = [-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4]
    fH2O_p = [ 0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516]
    comp = partial(np.interp, xp=FeH_p, fp=fH2O_p)
    FeH = np.linspace(FeHBinEdges_array[0,0], FeHBinEdges_array[-1,1], 100)
    fH2O = comp(FeH)
#    dfdFeH = (np.diff(fH2O)/(FeH[1]-FeH[0]), )

    fig, ax = plt.subplots()
    ax.plot(FeH, fH2O)
    ax.set_xlabel('[Fe/H]')
    ax.set_ylabel('Water fraction')
    fig.savefig(savePath + 'BB20.png')

    fH2OBinEdges = np.linspace(0,0.6,13)
    ISOdensity = np.zeros(len(fH2OBinEdges)-1)
    FeHBinEdges = np.append(FeHBinEdges_array[:,0], FeHBinEdges_array[-1,1]) #prob should redefine throguhout
    print(FeHBinEdges)
    for i in range(len(FeH)-1): #Need to skip last FeH due to fenceposting
        feh_j = np.where(FeH[i] >= FeHBinEdges)[0].max()
        iso_j = np.where(comp(FeH[i]) >= fH2OBinEdges)[0].max()
        #print(i,feh_j,iso_j)
        # bin index of composition of that point
        ISOdensity[iso_j] += massDensity[feh_j]*(FeH[1]-FeH[0])
    print(ISOdensity)

    fig, ax = plt.subplots()
    ax.bar(fH2OBinEdges[0:-1], ISOdensity/ISOdensity.max(), width=fH2OBinEdges[1]-fH2OBinEdges[0],align='edge')
    ax.set_xlabel('Water fraction')
    ax.set_ylabel('Number ISOs around Sun (arbitrary units)')
    fig.savefig(savePath+'fH2O.png')

class logNuSunDoubleExpPPP(TorchDistribution):
    """
    Note changes: Multiplier now doesnt't include bin width as true density is just density in space, not space and metalicity
    log_prob now calculated with N, meanR,meanmodz
    """

    arg_constraints = {
        'logNuSun': constraints.real,
        'a_R': constraints.real,
        'a_z':  constraints.real,
    }
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, logNuSun, a_R, a_z, R, modz, multiplier,
                 validate_args=None):
        """
        Distribution for (N,SumR,Summodz) for double exponential PPP model
        with APOGEE fields.
        logNuSun, a_R, a_z are Numbers or Tensors, R, z, multiplier are
        arrays
        multiplier is the array the true density is multiplied by before
        summing to get effective volume, equal to
        ln(10)*solidAngle*D**3*effSelFunct*muSpacing/5 -- now not containing binWidth
        arrays 0th index corresponding to field, 1st corresponding to
        mu grid points
        If I wanted to add batching, could it be done with multiplier etc?
        """
        self.logNuSun, self.a_R, self.a_z = broadcast_all(logNuSun, a_R, a_z)
        #makes parameters tensors and broadcasts them to same shape
        super().__init__(batch_shape=self.logNuSun.shape,
                         event_shape=torch.Size([3]),
                         validate_args=validate_args)
        self.R, self.modz, self.multiplier = torch.tensor(R), torch.tensor(modz), torch.tensor(multiplier)

    # Keeping things simple, not interfering with anything related to gradient by making them properties

    def nu(self):
        """
        Density array.
        The contribution to effVol from point i is nu_i*self.M_i
        Done to avoid many copies of same code, but needed as function as must change with latents
        Anything affecting log_prob containing latensts must be kept as tensors for derivative
        """
        return torch.exp(self.logNuSun-self.a_R*(self.R - R_Sun)-self.a_z*self.modz)

    def effVol(self):
        """
        Convenience method. Anything affecting log_prob containing latensts must be kept as tensors for derivative
        """
        return (self.multiplier*self.nu()).sum()

    def log_prob(self, value):
        """
        UNTESTED
        not designed to work with batches and samples
        value is[N,meanR,meanmodz]
        IMPORTANT: this is only log_p up to constant not dependent on latents
        """
        if self._validate_args:
            self._validate_sample(value)
        # all values in value need to be same type, so technically
        # tests N is a float, not int
        return (value[0]*(self.logNuSun - (value[1]-R_Sun)*self.a_R - value[2]*self.a_z)
                - self.effVol())

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sample should not be required")

def mu2D(mu):
    return 10**(-2+0.2*mu)

def D2mu(D):
    return 10 + 5*np.log10(D)

def calc_coords(apo):
    """makes mu, D, R, modz, solidAngles, gLon gLat, and galacticentric
    x, y, and z arrays, for ease of use
    units are kpc, and R and modz is for central angle of field.
    rows are for fields, columns are for mu values"""
    locations = apo.list_fields(cohort='all')
    Nfields = len(locations)
    # locations is list of ids of fields with at least completed cohort of
    #  any type, therefore some stars in statistical sample

    mu = arr(muGridParams)
    D = mu2D(mu)
    gLon = np.zeros((Nfields, 1))
    gLat = np.zeros((Nfields, 1))
    solidAngles = np.zeros((Nfields, 1))
    # This shape allows clever broadcasting in coord.SkyCoord
    for loc_index, loc in enumerate(locations):
        gLon[loc_index,0], gLat[loc_index,0] = apo.glonGlat(loc)
        solidAngles[loc_index,0] = apo.area(loc)*_DEGTORAD**2
    gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(GC_frame)
    x = gCentricCoords.x.value
    y = gCentricCoords.y.value
    z = gCentricCoords.z.value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)
    # check these are right values and shape - write tests!
    return mu, D, R, modz, solidAngles, gLon, gLat, x, y, z

def extract(S):
    gLon = S['GLON']
    gLat = S['GLAT']
    D = S['weighted_dist']/1000 # kpc
    #D_err =
    mu = 10 + 5*np.log10(D)
    gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(GC_frame)
    x = gCentricCoords.x.to(u.kpc).value
    y = gCentricCoords.y.to(u.kpc).value
    z = gCentricCoords.z.to(u.kpc).value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)

    FeH = S['FE_H'] #Check with Ted
    MgFe = S['MG_FE'] #check
    age = S['age_lowess_correct'] #check

    return mu, D, R, modz, gLon, gLat, x, y, z, FeH, MgFe, age

def get_apo():
    """
    """
    path =_DATADIR+'input_data/apodr16_csf.dat'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            apo = pickle.load(f)
    else:
        apo = apsel.apogeeCombinedSelect()
        with open(path, 'wb') as f:
            pickle.dump(apo, f)
    # maybe add del line here or later if I have memory problems
    return apo


def get_effSelFunct(binDict):
    path = (_DATADIR+'bins/'+binName(binDict)+'/effSelFunct.dat')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            effSelFunct = pickle.load(f)
    else:
        raise FileNotFoundError("Currently effSelFunct must be calculated seperately")
    return effSelFunct

def calc_multiplier(binDict, apo):
    effSelFunct = get_effSelFunct(binDict)
    mu, D, R, modz, solidAngles, *_ = calc_coords(apo)
    return (solidAngles*D**3
                  *(muStep)*effSelFunct*np.log(10)/5)

def isogridFieldAndFunct(label):
    """returns field in ischrone grids given my label. Add to if needed.
    if undefined in isochrone grid returns empty string. Beware of age
    vs logAge
    Thought: could additionally output a function which maps isogrid value to same scale/units as my values
    """
    field = 'UNEXPECTED'
    funct = lambda x: np.nan
    match label:
        case 'FeH':
            field = 'MH'
            funct = lambda x: x
        case 'MgFe':
            field = 'unused_in_isochrones'
        case 'age':
            field = 'logAge'
            funct = lambda x: 10**(x-9)
    return field, funct

def calc_isogrid_mask(binDict, isogrid, RG_only=True):
    if RG_only:
        mask = ((1<=isogrid['logg']) & (isogrid['logg']<3))
    else:
        mask = np.full(len(isogrid),True)

    for label, limits in binDict.items():
        field, funct = isogridFieldAndFunct(label)
        if field!='unused_in_isochrones':
            mask &= ((limits[0]<=funct(isogrid[field])) & (funct(isogrid[field])<limits[1]))
        else: pass
    return mask

def allStarFieldAndFunct(label):
    """returns field in allStar given my label. Add to if needed.
    if undefined returns empty string. Beware of age
    vs logAge type fields and units"""
    field = 'UNEXPECTED'
    funct = lambda x: np.nan
    match label:
        case 'D':
            field = 'weighted_dist'
            funct = lambda x: x/1000
        case 'mu':
            field = 'weighted_dist'
            funct = lambda x: D2mu(x/1000)
        case 'FeH':
            field = 'FE_H'
            funct = lambda x: x
        case 'MgFe':
            field = 'MG_FE'
            funct = lambda x: x
        case 'age':
            field = 'age_lowess_correct'
            funct = lambda x: max(min(x,13.8),0)
    return field, funct

def calc_allStarSample_mask(binDict, sample, in_mu_range_only=True):
    """
    Any subsample of allStar can be inputted (ie statSample) as long as fields are like allStar
    """
    if in_mu_range_only:
        field, funct = allStarFieldAndFunct('mu')
        mask = ((muMin<=funct(sample[field])) & (funct(sample[field])<muMax))
    else:
        mask = np.full(len(sample),True)

    for label, limits in binDict.items():
        field, funct = allStarFieldAndFunct(label)
        if field!='unused_in_allStar':
            mask &= ((limits[0]<=funct(sample[field])) & (funct(sample[field])<limits[1]))
        else: pass
    return mask


def get_allStar():
    """
    uses a pickle with options:
        rmcommissioning=True,
        main=True,
        exclude_star_bad=True,
        exclude_star_warn=True,
        use_astroNN_distances=True,
        use_astroNN_ages=True,
        rmdups=True
    """
    name = "dr16allStar.dat"
    path = _DATADIR+'input_data/'+name
    if os.path.exists(path):
        with open(path, 'rb') as f:
            allStar = pickle.load(f)
    else:
        print('WARNING: calculating allStar')
        allStar = apread.allStar(
            rmcommissioning=True,
            main=True,
            exclude_star_bad=True,
            exclude_star_warn=True,
            use_astroNN_distances=True,
            use_astroNN_ages=True,
            rmdups=True)
        with open(path, 'wb') as f:
            pickle.dump(allStar, f)
    assert len(allStar) == 211051
    return allStar


def get_allStar_statIndx():
    """
    returns both as whenever statIndx is needed, allStar is too
    """
    allStar = get_allStar()

    statIndxPath = _DATADIR+'input_data/dr16statIndx.dat'
    if os.path.exists(statIndxPath):
        with open(statIndxPath, 'rb') as f:
            statIndx = pickle.load(f)
    else:
        print('WARNING: calculating statIndx')
        apo = load_apo()
        statIndx = apo.determine_statistical(allStar)
        with open(path, 'wb') as f:
            pickle.dump(statIndx, f)
    assert np.count_nonzero(statIndx)==165768
    return (allStar, statIndx)

### DEPRECIATED CODE ###

class oneBinDoubleExpPPP(TorchDistribution):
    """
    """

    arg_constraints = {
        'logA': constraints.real,
        'a_R': constraints.real,
        'a_z':  constraints.real,
    }
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, logA, a_R, a_z, R, modz, multiplier,
                 validate_args=None):
        """
        Distribution for (N,SumR,Summodz) for double exponential PPP model
        with APOGEE fields.
        Allows batching of parameters, but only one bin at a time.
        logA, a_R, a_z are Numbers or Tensors, FeHBinEdges, R, z, jac are
        arrays
        multiplier is the array the true density is multiplied by before
        summing to get effective volume, equal to
        ln(10)*solidAngle*D**3*binWidth*effSelFunct*muSpacing/5
        arrays 0th index corresponding to field, 1st corresponding to
        mu grid points
        FeHBinEdges doesn't appear to be used
        """
        warnings.warn("DEPRECIATED: oneBinDoubleExpPPP")
        self.logA, self.a_R, self.a_z = broadcast_all(logA, a_R, a_z)
        #makes parameters tensors and broadcasts them to same shape
        super().__init__(batch_shape=self.logA.shape,
                         event_shape=torch.Size([3]),
                         validate_args=validate_args)
        self.R, self.modz, self.multiplier = torch.tensor(R), torch.tensor(modz), torch.tensor(multiplier)
#        trueDens = np.exp(self.logA.item())*self.norm_nu_array(R,modz)
#        self._effVol = (trueDens*multiplier).sum()
#        print(f"Instantiating oneBin with: {logA}, {a_R}, {a_z}, {self.effVol}")


# Keeping things simple, not interfering with anything related to gradient by making them properties

    def norm_nu(self):
        """
        Normalised density array.
        The contribution to effVol from point i is nu_norm_i*exp(self.logA)*self.M_i
        Done to avoid many copies of same code, but needed as function as must change with latents
        Anything affecting log_prob containing latensts must be kept as tensors for derivative
        """
        return (self.a_R**2 * self.a_z
                    *torch.exp(-self.a_R*self.R
                               -self.a_z*self.modz)/(4*torch.pi))

    def effVol(self):
        """
        Convenience method. Anything affecting log_prob containing latensts must be kept as tensors for derivative
        """
        return (self.multiplier*torch.exp(self.logA)*self.norm_nu()).sum()

    def log_prob(self, value):
        """
        UNTESTED
        not designed to work with batches and samples
        value is[N,sumR,summodz]
        IMPORTANT: this is only log_p up to constant not dependent on latents
        For now explicit to be safe, may move effvol calc elsewhere to avoid defining twice
        """
        if self._validate_args:
            self._validate_sample(value)
        # all values in value need to be same type, so technically
        # tests N is a float, not int
        return (value[0]*(self.logA+2*torch.log(self.a_R)+torch.log(self.a_z))
                - value[1]*self.a_R - value[2]*self.a_z
                - self.effVol())

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sample should not be required")



class FeHBinnedDoubleExpPPP(TorchDistribution):
    """
    DEPRECIATED - contains far too much to be useful for fitting, replaced by oneBinDoubleExpPPP

    This is the distribution for [N, sumR, summodz] for an exponential disk PPP of stars uniform in
    metallicity in bins, which are the
    only data quantities which appear in the likelihood function.

    Inheriting TorchDistribution means I can use torch.distributions.Distribution's __init__,
    which automatically sorts batch_shape, event_shape and validation of arguments, and methods
    from TorchDistribution and TorchDistributionMixin such as shape, to_event.

    since definition of log_p, a batch size of one is assumed in this code in the hope that
    plate sorts batching automatically

    Currently only doing things which have to be tensors as tensors eg inputs and outputs to distro
    Others are done as np arrays
    """

    arg_constraints = {
        'logA': constraints.real,
        'a_R': constraints.real,
        'a_z':  constraints.real,
    }
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, FeHBinEdges, logA, a_R, a_z, validate_args=None):
        """
        FeHBinEdges is tensor with final dimention length 2, representing lower and upper bound
        of bins. logA, a_R, a_z are parameters, can be entered as tensors to get batching.
        validate_args is used by torch.distributions.Distribution __init__ to decide whether or not
        to check parameters and sample values match constraints. True/False means check/don't check,
        None means default behavior.
        """
        warnings.warn("DEPRECIATED: FeHBinnedDoubleExpPPP")
        print("PPP has been initialised!")
        for value, name in [(FeHBinEdges,"FeHBinEdges"),(logA,"logA"),(a_R,"a_R"),(a_z,"a_z")]:
            if not (isinstance(value,torch.Tensor) or isinstance(value,Number)):
                raise ValueError(name+" must be either a Number or a torch.Tensor")
        lowerEdges, upperEdges, self._logA, self._a_R, self._a_z = broadcast_all(FeHBinEdges[...,0], FeHBinEdges[...,1], logA, a_R, a_z)
        self._FeHBinEdges = torch.stack((lowerEdges,upperEdges), lowerEdges.dim()) #this gives wrong shape when others areentered as 1-element tensors
        super().__init__(batch_shape=self.logA.shape, event_shape=torch.Size([3]), validate_args=validate_args)

        if os.path.exists(_ROOTDIR+'sav/apodr16_csf.dat'):
            with open(_ROOTDIR+'sav/apodr16_csf.dat', 'rb') as f:
                self._apo = pickle.load(f)
        else:
            self._apo = apsel.apogeeCombinedSelect()
            with open(_ROOTDIR+'sav/apodr16_csf.dat', 'wb') as f:
                pickle.dump(self.apo, f)
        # maybe add delete line here if I have memory problems

        muMin = 0.0
        muMax = 15 # CHECK THIS against allStar - better to exclude fringe datapoints than have all data in nearest three bins - plot mu distribution
        muDiff = 0.1
        muGridParams = (muMin, muMax, int((muMax-muMin)//muDiff)) # (start,stop,size)
        self._mu, self._D, self._R, self._modz, self._solidAngles = self.makeCoords(muGridParams)
        self._effSelFunct = self.makeEffSelFunct()


    @property
    def logA(self):
        """A """
        return self._logA

    @property
    def a_R(self):
        """ d"""
        return self._a_R

    @property
    def a_z(self):
        """t """
        return self._a_z

    @property
    def FeHBinEdges(self):
        """returns bin edges. Writing as property reduces risk of them accidentlly being changed after object is created"""
        return self._FeHBinEdges

    @property
    def apo(self):
        """apo"""
        return self._apo

    @property
    def mu(self):
        """mu grid"""
        return self._mu

    @property
    def D(self):
        """D grid"""
        return self._D

    @property
    def R(self):
        """R grid"""
        return self._R

    @property
    def modz(self):
        """modz grid"""
        return self._modz

    @property
    def solidAngles(self):
        """(Nfields,1) grid of solid angles of fields in rad**2"""
        return self._solidAngles

    @property
    def effSelFunct(self):
        """effSelFunct grid"""
        return self._effSelFunct


    def makeCoords(self, muGridParams):
        """makes mu, D, R, modz tensors, for ease of use
        units are kpc, and R and modz is for central angle of field"""
        locations = self.apo.list_fields(cohort='all')
        # locations is list of indices of fields with at least completed cohort of any type, therefore some stars in statistical sample
        # default value of cohort is "short", which only includes fields with completed short cohort, which tbh probably will always be first cohort to be completed
        Nfields = len(locations)

        mu = np.linspace(*muGridParams)
        D = 10**(-2+0.2*mu) #kpc
        gLon = np.zeros((Nfields, 1))
        gLat = np.zeros((Nfields, 1))
        solidAngles = np.zeros((Nfields, 1))
        for loc_index, loc in enumerate(locations):
            gLon[loc_index,0], gLat[loc_index,0] = self.apo.glonGlat(loc)
            solidAngles[loc_index,0] = self.apo.area(loc)*_DEGTORAD**2
        # gLon, gLat = np.split(np.array([self.apo.glonGlat(loc) for loc in locations]),2,1)
        # makes gLon and gLat as (Nfield,1) arrays
        gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
        # check this has right layout
        gCentricCoords = gCoords.transform_to(coord.Galactocentric)
        # check this looks right
        R = np.sqrt(gCentricCoords.x.value**2 + gCentricCoords.y.value**2)
        modz = np.abs(gCentricCoords.z.value) # mixing astropy units and torch.tensors casues problems
        # check these are right values and shape - write tests!
        return mu, D, R, modz, solidAngles

    def makeEffSelFunct(self, rewrite=False):
        """UNTESTED - ASSUMES ONE BIN ONLY
        if EffSelFunct tensor exists for bin, loads it. If not, it calculates it
        in future may Return S[bin_index, loc_index, mu_index]
        for now assumes one bin
        Currently uses samples - chat to Ted and change to using whole isochrone (currently can't see any code on github that can take 'weights' argument)
        """
        filePath = (_ROOTDIR+"sav/EffSelFunctGrids/" +
                    '_'.join([str(self.FeHBinEdges[0].item()), str(self.FeHBinEdges[1].item()), str(self.mu[0]),
                              str(self.mu[-1]), str(len(self.mu))])
                    + ".dat")
        if os.path.exists(filePath) and (not rewrite):
            with open(filePath, 'rb') as f:
                effSelFunct = pickle.load(f)
        else:
            raise NotImplementedError("Currently effSelFunct must be calculated seperately")
            #effSelFunct = effSelFunct_calculate(self)
            #with open(filePath, 'wb') as f:
            #    pickle.dump(effSelFunct, f)

        return effSelFunct


    @property
    def effectiveVolume(self):
        """
        UNTESTED - UNFINISHED - ASSUMES ONE BIN
        Just multiple density, jacobian and effsel grids, then sum
        Assumes grid uniform in mu and one bin
        """ #may need .item on tensors here
        trueDens = self.a_R.item()**2 * self.a_z.item() * np.exp(self.logA.item()-self.a_R.item()*self.R-self.a_z.item()*self.modz)/(4*np.pi)
        jac = self.D**3 *np.log(10)*self.solidAngles*(self.FeHBinEdges[1].item()-self.FeHBinEdges[0].item())/5
        return ((self.mu[1]-self.mu[0])*trueDens*jac*self.effSelFunct).sum()

    def log_prob(self, value):
        """
        UNTESTED
        make sure works with batches and samples
        value[...,i] will be [N,sumR,summodz] for i=0,1,2, but im not sure if broadcasting will match 
        batch_shapped logA etc to batch indices of value
        """
        if self._validate_args:
            self._validate_sample(value)
        # can I check for int here? all values in value need to be same type
        return (value[...,0]*(self.logA+2*torch.log(self.a_R)+torch.log(self.a_z))
                - value[...,1]*self.a_R - value[...,2]*self.a_z - self.effectiveVolume)
        #IMPORTANT: this is only log_p up to constant not dependent on latents- check if this is ok - full log_p requires sumlogD and sumlogEffSelFucnct(D), which will be slower


    def sample(self, sample_shape=torch.Size()):
        """
        May never implement, as not needed for fitting.
        Should sample event (N, SumR, SumModz) in whatever sample shape requested
        """
        pass

