#from numbers import Number
import pickle
import os
#import datetime
import warnings

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
#import matplotlib.pyplot as plt


_DEGTORAD = torch.pi/180
_ROOTDIR = "/home/sjoh4701/APOGEE/iso-apogee/"

GC_frame = coord.Galactocentric() #adjust parameters here if needed
z_Sun = GC_frame.z_sun.value # .value removes unit, which causes problems with pytorch
R_Sun = np.sqrt(GC_frame.galcen_distance.value**2 - z_Sun**2)

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
        The contribution to effVol from point i is nu_norm_i*exp(self.logA)*self.M_i
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


def makeCoords(muGridParams, apo):
    """makes mu, D, R, modz, solidAngles, gLon gLat, and galacticentric
    x, y, and z arrays, for ease of use
    units are kpc, and R and modz is for central angle of field.
    rows are for fields, columns are for mu values"""
    locations = apo.list_fields(cohort='all')
    Nfields = len(locations)
    # locations is list of ids of fields with at least completed cohort of
    #  any type, therefore some stars in statistical sample

    mu = np.linspace(*muGridParams)
    D = 10**(-2+0.2*mu) #kpc
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


def load_apo():
    """
    """
    if os.path.exists(_ROOTDIR+'sav/apodr16_csf.dat'):
        with open(_ROOTDIR+'sav/apodr16_csf.dat', 'rb') as f:
            apo = pickle.load(f)
    else:
        apo = apsel.apogeeCombinedSelect()
        with open(_ROOTDIR+'sav/apodr16_csf.dat', 'wb') as f:
            pickle.dump(apo, f)
    # maybe add del line here or later if I have memory problems
    return apo

def calculate_R_modz_multiplier(FeHBinEdges, muGridParams):
    """
    returns tuple (R, modz, multiplier) for unpacking and inputting to
    oneBinDoubleExpPPP
    Strange looking structure (since R, modz is returned by makeCoords) for easy fitting
    """
    apo = load_apo()

    mu, D, R, modz, solidAngles, *_ = makeCoords(muGridParams, apo)

    ESFpath = (_ROOTDIR+"sav/EffSelFunctGrids/" +
                '_'.join([str(float(FeHBinEdges[0])),
                          str(float(FeHBinEdges[1])),
                          str(float(mu[0])),
                          str(float(mu[-1])),
                          str(len(mu))])
                + ".dat")
    if os.path.exists(ESFpath):
        with open(ESFpath, 'rb') as f:
            effSelFunct = pickle.load(f)
    else:
        raise FileNotFoundError("Currently effSelFunct must be calculated seperately")

    multiplier = (solidAngles*D**3
                  *(mu[1]-mu[0])*effSelFunct*np.log(10)/5)
    return (R, modz, multiplier)

def load_allStar():
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
    path = _ROOTDIR+'sav/'+name
    if os.path.exists(path):
        with open(path, 'rb') as f:
            allStar = pickle.load(f)
    else:
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


def load_statIndx():
    """
    uses a pickle of load_allStar
    """
    name = "dr16statIndx.dat"
    path = _ROOTDIR+'sav/'+name
    if os.path.exists(path):
        with open(path, 'rb') as f:
            statIndx = pickle.load(f)
    else:
        apo = load_apo()
        allStar = load_allStar()
        statIndx = apo.determine_statistical(allStar)
        with open(path, 'wb') as f:
            pickle.dump(statIndx, f)
    assert np.count_nonzero(statIndx)==165768
    return statIndx

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

