from numbers import Number
import os

assert os.getenv('RESULTS_VERS')=='l33' # apogee should automatically use dr16 as long as this is correct

import apogee.select as apsel
import apogee.tools.read as apread
import isochrones as iso

import torch
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.distributions.util import broadcast_all

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import astropy.coordinates as coord
import astropy.units as u


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

        if os.path.exists('../sav/apodr16_csf.dat'):
            with open('../sav/apodr16_csf.dat', 'rb') as f:
                self._apo = pickle.load(f)
        else:
            self._apo = apsel.apogeeCombinedSelect()
            with open('../sav/apodr16_csf.dat', 'wb') as f:
                pickle.dump(self.apo, f)

        muMin = 0.0
        muMax = 16.5 # CHECK THIS against allStar
        muDiff = 0.1
        muGridParams = (muMin, muMax, int((muMax-muMin)//muDiff)) # (start,stop,size)
        self._mu, self._D, self._R, self._modz = self.makeCoords(muGridParams)
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
        gLon = np.zeros((Nfields, 1))
        gLat = np.zeros((Nfields, 1))
        for i_loc, loc in enumerate(locations):
            gLon[i_loc,1], gLat[i_loc,1] = self.apo.glonGlat(loc)
        gLon, gLat = np.split(np.array([list(self.apo.glonGlat(loc)) for loc in locations]),2,axis=1) #makes gLon and gLat as (Nfield,1) arrays
        gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc)
        # check this has right layout
        gCentricCoords = gCoords.transform_to(coord.Galactocentric)
        # check this looks right
        R = torch.tensor(np.sqrt(gCentricCoords.x**2, gCentricCoords.y**2))
        modz = torch.tensor(np.abs(gCentricCoords.x))
        # check these are right values and shape - write tests!
        return mu, D, R, modz

    def makeEffSelFunct(self):
        """if EffSelFunct tensor exists for bin, loads it. If not, it calculates it"""
        

    def EffectiveVolume(self, optionBool):
        """
        UNTESTED - ASSUMES ONE BIN ONLY
        parallelisabel?
        Ted's isochrones are used with isogrid, an astropy.Table, masks made by isogrid>value which is probably np.arraylike
        """
        if os.path.exists('../sav/apodr16_csf.dat'):
            with open('../sav/apodr16_csf.dat', 'rb') as f:
                apo = pickle.load(f)
        else:
            raise ValueError("precalculate combined selection function")
        locations = apo.list_fields(cohort='all')
        # locations is list of fields with at least completed cohort of any type, therefore some stars in statistical sample
        # default value of cohort is "short", which only includes fields with completed short cohort, which tbh probably will always be first cohort to be completed
        print("total number of fields: " + len(locations))

        Neffsamp = 600 #tune this, or change to sampling entire isochrone
        isogrid = iso.newgrid()
        mask = ((isogrid['logg'] > 1) & (isogrid['logg'] < 3) & (isogrid['MH'] > self.FeHBinEdges[0])
                                                              & (isogrid['MH'] <= self.FeHBinEdges[1]))
        effsel_samples = iso.sampleiso(Neffsamp, isogrid[mask], newgrid=True) #why new grid?

        apof = apsel.apogeeEffectiveSelect(apo, dmap3d=dmap, MH=effsel_samples['Hmag'],
                                           JK0=effsel_samples['Jmag']-effsel_samples['Ksmag'])
        if optionBool: #calculated myself
            mu_grid = 
            sum = 0
            for loc in locations:
                
        else: # calculated with scipy


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
        integral = self.EffectiveVolume() #play around with defining _Effective volume as a property, or a method (ie calculated once every time instance made, or calculated every time it is used) - may effect speed
        return value[...,0]*(self.logA+2*torch.log(self.a_R)+torch.log(self.a_z)) - value[...,1]*self.a_R - value[...,2]*self.a_z - integral
        #IMPORTANT: this is only log_p up to constant not dependent on latents- check if this is ok - full log_p requires sumlogD and sumlogEffSelFucnct(D), which will be slower


    def sample(self, sample_shape=torch.Size()):
        """
        May never implement, as not needed for fitting.
        Should sample event (N, SumR, SumModz) in whatever sample shape requested
        """
        pass


def model(FeHBinEdges, sums):
    """
    FeHBinEdges mark edges of metallicity bins being used, in form of 2*nbins tensor with [lowerEdge, upperEdge] for each bin
    sums is 3*nbins tensor with [N,sumR,summodz] for each bin
    """
    with pyro.plate('bins', len(FeHBinEdges)-1): # needs new bin definition: do something like with plate as 
        logA = pyro.sample('logA', dist.Normal(...))
        a_R = pyro.sample('a_R', dist.Normal(...))
        a_z = pyro.sample('a_z', dist.Normal(...))
        return pyro.sample('obs', MyDist(FeHBinEdges, logA, a_R, a_z, validate_args=True), obs=sums)

# def guide or autoguide




### main ###


    
    
    
    
    
    
    
    #optimiser = Adam() #look at parameters
    #svi = SVI(model, guide, optimiser, loss=Trace_ELBO())
    
    #losses = []
    #for step in range(1000):
    #    loss = svi.step(FeHBinEdges, sums)
    #    losses.append(loss)
    #    if step%100==0:
    #        print(f'Loss = {loss}')
    #extract data
