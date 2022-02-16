import torch
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.distributions.util import broadcast_all

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class FeHBinnedDoubleExpPPP(TorchDistribution):
    """
    This is the distribution for [N, sumR, summodz] for an exponential disk PPP of stars uniform in
    metallicity in bins, which are the
    only data quantities which appear in the likelihood function. Default is for one bin, but can
    be batched for multiple bins.

    Inheriting TorchDistribution means I can use torch.distributions.Distribution's __init__,
    which automatically sorts batch_shape, event_shape and validation of arguments, and methods
    from TorchDistribution and TorchDistributionMixin such as shape, to_event.
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
        logA, a_R, a_z are parameters, can be entered as tensors all of same size to get batch.
        validate_args is used by torch.distributions.Distribution __init__ to decide whether or not
        to check parameters and sample values match constraints. True/False means check/don't check,
        None means default behavior.

        may need to add bin edges somehow - make it work with batches
        """
        self._FeHBinEdges = FeHBinEdges
        self.logA, self.a_R, self.a_z = broadcast_all(logA, a_R, a_z)
        # broadcasts tensors so they are same shape. Ideally have way of broadcasting all scalar parameters to (n,) if n given by bins
        # make checks like parameters all have same length, work out what to do with batch and event shape - may need to define batch_shape
        #super().__init__(batch_shape=, event_shape=, validate_args=validate_args)

    @property
    def FeHBinEdges(self):
        """returns bin edges. Writing as property reduces risk of them accidentlly being changed after object is created"""
        return self._FeHBinEdges


    def log_prob(self, value):
        """
        make sure works with batches and samples
        value[...,i] will be [N,sumR,summodz] for i=0,1,2, but im not sure if broadcasting will match 
        batch_shapped logA etc to batch indices of value
        """
        if self._validate_args:
            self._validate_sample(value)
        #define integral, sum over fields of integral over distance of rate function
        return value[...,0]*(self.logA+2*torch.log(self.a_R)+torch.log(self.a_z)) - value[...,1]*self.a_R - value[...,2]*self.a_z - integral
        #IMPORTANT: this is only log_p up to constant not dependent on latents- check if this is ok - full log_p requires sumlogD and sumlogEffSelFucnct(D), which will be slower

    def sample(self, sample_shape=torch.Size()):
        """
        May never implement, as not needed for fitting
        """
        pass


def model(FeHBinEdges, sums):
    """
    FeHBinEdges mark edges of metallicity bins being used, in form of 2*nbins tensor with [lowerEdge, upperEdge] for each bin
    sums is 3*nbins tensor with [N,sumR,summodz] for each bin
    """
    with pyro.plate('bins', len(FeHBinEdges)-1): # needs new bin definition
        logA = pyro.sample('logA', dist.Normal(...))
        a_R = pyro.sample('a_R', dist.Normal(...))
        a_z = pyro.sample('a_z', dist.Normal(...))
        return pyro.sample('obs', MyDist(FeHBinEdges, logA, a_R, a_z, validate_args=True), obs=sums)

# def guide or autoguide

# main

#optimiser = Adam() #look at parameters
#svi = SVI(model, guide, optimiser, loss=Trace_ELBO())

#losses = []
#for step in range(1000):
#    loss = svi.step(FeHBinEdges, sums)
#    losses.append(loss)
#    if step%100==0:
#        print(f'Loss = {loss}')
#extract data
