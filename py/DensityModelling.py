import torch
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.distributions.utils import broadcast_all

from pyro.distributions.torch_distribution import TorchDistribution

class MyDist(TorchDistribution): # think of better name
    """
    Write something here

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
    has_rsample = 

    def __init__(self, logA, a_R, a_z, validate_args=None):
        self.logA, self.a_R, self.a_z = broadcast_all(logA, a_R, a_z) # probably converts all to tensors
        # make checks like parameters all have same length, define batch_shape
        super().__init__(batch_shape=batch_shape, event_shape=, validate_args=validate_args)

    def log_prob(self, value):
        """
        make sure works with batches and samples
        value[...,i] will be [N,sumR,summodz] for i=0,1,2, but im not sure if broadcasting will match 
        batch_shapped logA etc to batch indices of value
        """
        if self._validate_args:
            self._validate_sample(value)
        #define integral, sum over fields of integral over distance of rate function
        return value[...,0]*(self.logA+2*torch.log(self.a_R)+torch.log(self.a_z) - value[...,1]*self.a_R - value[...,2]*self.a_z - integral

    def rsample(self, sample_shape=torch.Size()):
        """I don't know how to do this, or if reparatrimisation is possible"""
            pass


def model(FeHBinEdges, sums):
    with pyro.plate('bins', len(FeHBinEdges)-1):
        logA = pyro.sample('logA', dist.Normal(...))
        a_R = pyro.sample('a_R', dist.Normal(...))
        a_z = pyro.sample('a_z', dist.Normal(...))
        return pyro.sample('obs', MyDist, obs=sums)
