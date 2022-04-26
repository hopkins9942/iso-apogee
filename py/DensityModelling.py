from numbers import Number
import pickle
import os
from functools import partial

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

import multiprocessing
import schwimmbad


_DEGTORAD = torch.pi/180
_ROOTDIR = "/home/sjoh4701/APOGEE/iso-apogee/"
_NCORES = 24 # needs to be set explicitly, otherwise multiprocessing tries to use 48 - I think

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

        mu = torch.linspace(*muGridParams)
        D = 10**(-2+0.2*mu) #kpc
        gLon = torch.zeros((Nfields, 1))
        gLat = torch.zeros((Nfields, 1))
        solidAngles = torch.zeros((Nfields, 1))
        for loc_index, loc in enumerate(locations):
            #gLon[loc_index,1], gLat[loc_index,1] = self.apo.glonGlat(loc)
            solidAngles[loc_index,0] = self.apo.area(loc)*_DEGTORAD**2
        gLon, gLat = np.split(np.array([self.apo.glonGlat(loc) for loc in locations]),2,1)
        # makes gLon and gLat as (Nfield,1) arrays
        gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
        # check this has right layout
        gCentricCoords = gCoords.transform_to(coord.Galactocentric)
        # check this looks right
        R = torch.tensor(np.sqrt(gCentricCoords.x**2 + gCentricCoords.y**2))
        modz = torch.tensor(np.abs(gCentricCoords.z))
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
            locations = self.apo.list_fields(cohort='all')

            effSelFunct = torch.zeros(len(locations),len(self.mu))

            #Neffsamp = 600 #tune this, or change to sampling entire isochrone
            isogrid = iso.newgrid()
            # newgrid ensures means it uses new isochrones. I should either rewrite isochrones.py, maybe with MIST isochrones, or at least fully understand it
            mask = ((isogrid['logg'] > 1) & (isogrid['logg'] < 3)
                    & (isogrid['MH'] >  self.FeHBinEdges[0].item())
                    & (isogrid['MH'] <= self.FeHBinEdges[1].item()))
            #effsel_samples = iso.sampleiso(Neffsamp, isogrid[mask], newgrid=True)
            # newgrid ensures means it uses new isochrones. I should either rewrite isochrones.py, maybe with MIST isochrones, or at least fully understand it

            dmap = mwdust.Combined19(filter='2MASS H')
            # see Ted's code for how to include full grid
            #apof = apsel.apogeeEffectiveSelect(self.apo, dmap3d=dmap, MH=effsel_samples['Hmag'], JK0=effsel_samples['Jmag']-effsel_samples['Ksmag'])
            #this giving me an error caused by a sample apparently with JK0 less than minimum colour bin limit
            apof = apsel.apogeeEffectiveSelect(self.apo, dmap3d=dmap,
                                               MH=isogrid[mask]['Hmag'],
                                               JK0=isogrid[mask]['Jmag']-isogrid[mask]['Ksmag'],
                                               weights=isogrid[mask]['weights'])

            #for loc_index, loc in enumerate(locations):
            #    effSelFunct[loc_index,:] = apof(loc, np.array(self.D))

            effSelFunct_mapper = partial(effSelFunct_helper, apof, self.D, locations)
#            with multiprocessing.Pool(2) as p:
#                print("starting multiprocessing")
#                effSelFunct = torch.tensor(np.array(list(tqdm.tqdm(p.map(effSelFunct_mapper, range(len(locations))), total=len(locations)))))

            p = multiprocessing.Pool(2)
            print("trying to mp")
            effSelFunct = torch.tensor(np.array(list(tqdm.tqdm(p.map(effSelFunct_mapper, range(len(locations))), total=len(locations)))))
            p.close()

            # this arcane series of tensors, arrays, lists and maps is because 
            # a list is because tensors are best constructed out of a single
            # array rather than a list of arrays, and neither np.array nor
            # torch.tensor know how to deal directly with a map object


            with open(filePath, 'wb') as f:
                pickle.dump(effSelFunct, f)

        return effSelFunct

    @property
    def effectiveVolume(self):
        """
        UNTESTED - UNFINISHED - ASSUMES ONE BIN
        Just multiple density, jacobian and effsel grids, then sum
        Assumes grid uniform in mu and one bin
        """
        return ((self.a_R**2)*self.a_z*(self.FeHBinEdges[1]-self.FeHBinEdges[0])*torch.log(10)
                *(self.mu[1]-self.mu[0])*self.solidAngles*(self.D**3)*self.effSelFunct
                *torch.exp(self.logA-self.a_R*self.R-self.a_z*self.modz)/(20*torch.pi)).sum()

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



def effSelFunct_helper(apof, D, locations, i):
    """
    Needed as multiprocessed functions need to be defined at top level
    """
    print(i)
    return apof(locations[i], D)


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
    with multiprocessing.Pool(24) as p:
        print(p._processes)
    with schwimmbad.MultiPool(24) as p:
        print(p._processes)


    FeHBinEdges = torch.tensor([-0.5,0])
    distro = FeHBinnedDoubleExpPPP(FeHBinEdges, 0, 1, 1)
    ESF = distro.effSelFunct
    effVol = distro.effectiveVolume
    log_p = distro.log_prob(1,1,1)
    print(ESF)
    print(effVol)
    print(log_p)
    print(distro.R.size())
    print(distro.z.size())
    print(distro.effSelFunct.size())
    
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
