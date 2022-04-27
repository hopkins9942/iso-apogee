from numbers import Number
import pickle
import os

assert os.getenv('RESULTS_VERS')=='l33' # apogee should automatically use dr16 as long as this is correct

import apogee.select as apsel
import apogee.tools.read as apread
import isochrones as iso
import mwdust
import tqdm

import numpy as np
import torch
import pyro
import pyro.distributions as distributions
import pyro.distributions.constraints as constraints
from pyro.distributions.util import broadcast_all
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import astropy.coordinates as coord
import astropy.units as u

import corner



_DEGTORAD = torch.pi/180
_ROOTDIR = "/home/sjoh4701/APOGEE/iso-apogee/"
_NCORES = 12 # needs to be set explicitly, otherwise multiprocessing tries to use 48 - I think

class FeHBinnedDoubleExpPPP(TorchDistribution):
    """
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
        for value, name in [(FeHBinEdges,"FeHBinEdges"),(logA,"logA"),(a_R,"a_R"),(a_z,"a_z")]:
            if not (isinstance(value,torch.Tensor) or isinstance(value,Number)):
                raise ValueError(name+" must be either a Number or a torch.Tensor")
        lowerEdges, upperEdges, self.logA, self.a_R, self.a_z = broadcast_all(FeHBinEdges[...,0], FeHBinEdges[...,1], logA, a_R, a_z)
        self._FeHBinEdges = torch.stack((lowerEdges,upperEdges), lowerEdges.dim())
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
            #gLon[loc_index,1], gLat[loc_index,1] = self.apo.glonGlat(loc)
            solidAngles[loc_index,0] = self.apo.area(loc)*_DEGTORAD**2
        gLon, gLat = np.split(np.array([self.apo.glonGlat(loc) for loc in locations]),2,1)
        # makes gLon and gLat as (Nfield,1) arrays
        gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
        # check this has right layout
        gCentricCoords = gCoords.transform_to(coord.Galactocentric)
        # check this looks right
        R = np.sqrt(gCentricCoords.x**2 + gCentricCoords.y**2)
        modz = np.abs(gCentricCoords.z)
        # check these are right values and shape - write tests!
        return mu, D, R, modz, solidAngles

    def makeEffSelFunct(self, rewrite=False):
        """UNTESTED - ASSUMES ONE BIN ONLY
        if EffSelFunct tensor exists for bin, loads it. If not, it calculates it
        in future may Return S[bin_index, loc_index, mu_index]
        for now assumes one bin
        Currently uses samples - chat to Ted and change to using whole isochrone (currently can't see any code on github that can take 'weights' argument)
        """
        filePath = (_ROOTDIR+"sav/EffSelGrids/" +
                    '_'.join([str(self.FeHBinEdges[0]), str(self.FeHBinEdges[1]), str(self.mu[0]),
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
        return ((self.a_R**2)*self.a_z*(self.FeHBinEdges[1]-self.FeHBinEdges[0])*np.log(10)
                *(self.mu[1]-self.mu[0])*self.solidAngles*(self.D**3)*self.effSelFunct
                *np.exp(self.logA-self.a_R*self.R-self.a_z*self.modz)/(20*np.pi)).sum()

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




def model(FeHBinEdges, sums=None):
    """
    FeHBinEdges mark edges of metallicity bins being used, in form of 2*nbins tensor with [lowerEdge, upperEdge] for each bin
    sums is 3*nbins tensor with [N,sumR,summodz] for each bin

    FOR NOW ASSUMES ONE BIN
    """
    #with pyro.plate('bins', len(FeHBinEdges-1): # needs new bin definition: do something like with plate as 
    #    logA = pyro.sample('logA', dist.Normal(...))
    #    a_R = pyro.sample('a_R', dist.Normal(...))
    #    a_z = pyro.sample('a_z', dist.Normal(...))
    #    return pyro.sample('obs', MyDist(FeHBinEdges, logA, a_R, a_z, validate_args=True), obs=sums)

    logA = pyro.sample('logA', distributions.Normal(20, 10)) # tune these
    a_R = pyro.sample('a_R', distributions.Normal(5, 3))
    a_z = pyro.sample('a_z', distributions.Normal(1, 0.5))
    return pyro.sample('obs', FeHBinnedDoubleExpPPP(FeHBinEdges, logA, a_R, a_z), obs=sums)

# def guide or autoguide

mvn_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)


### main ###

if __name__ == "__main__":
    print(os.cpu_count())

    FeHBinEdges = torch.tensor([-0.5,0])
    distro = FeHBinnedDoubleExpPPP(FeHBinEdges, 0, 1, 1)
    ESF = distro.effSelFunct
    effVol = distro.effectiveVolume
    log_p = distro.log_prob(1,1,1)
    print(ESF)
    print(effVol)
    print(log_p)
    print(distro.R.shape())
    print(distro.z.shape())
    print(distro.effSelFunct.shape())
    
    optimiser = Adam({"lr": 0.02}) # tune, and look at other parameters
    svi = SVI(model, mvn_guide, optimiser, loss=Trace_ELBO())
    

    mock_sums = (10**10)*torch.tensor([1,5,0.3])

    losses = []
    for step in range(1000):
        loss = svi.step(FeHBinEdges, mock_sums)
        losses.append(loss)
        if step%100==0:
            print(f'Loss = {loss}')
    
    with pyro.plate("samples",500,dim=-1):
        samples = mvn_guide(FeHBinEdges) #samples latent space, accessed with samples["logA"] etc

    fig = corner.corner(samples)

    fig.savefig("/data/phys-galactic-isos/sjoh4701/APOGEE/mock_latents.png")
