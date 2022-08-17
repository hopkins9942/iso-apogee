import sys
import numpy as np

import torch
import pyro
import pyro.distributions.constraints as constraints
from pyro.distributions.util import broadcast_all
from pyro.distributions.torch_distribution import TorchDistribution

from myUtils import binsToUse, arr, R_Sun, z_Sun
import pickleGetters

def main():
    


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

def model(R_modz_multiplier, data=None):
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

    logNuSun = pyro.sample('logNuSun', distributions.Normal(10, 4)) # tune these
    a_R = pyro.sample('a_R', distributions.Normal(0.25, 0.01))
    a_z = pyro.sample('a_z', distributions.Normal(2, 1))
    return pyro.sample('obs', dm.logNuSunDoubleExpPPP(logNuSun, a_R, a_z, *R_modz_multiplier), obs=data)

if __name__=='__main__':
    main()
