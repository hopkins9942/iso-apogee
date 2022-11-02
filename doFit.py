import sys
import os
import datetime
import pickle

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

import matplotlib.pyplot as plt
import corner

import myUtils
import pickleGetters

# A lot here could be neatened
# put multiplier in with coords? And only compute/return needed coords

def main():
    print(f"Starting! {datetime.datetime.now()}")

    apo = pickleGetters.get_apo()
    
    binNum = int(sys.argv[1])

    binList = myUtils.binsToUse

    binDict = binList[binNum]
    
    MAP = False
    
    if MAP:
        binPath = os.path.join(myUtils.clusterDataDir, 'bins-MAP', myUtils.binName(binDict))
    else:
        binPath = os.path.join(myUtils.clusterDataDir, 'bins', myUtils.binName(binDict))
    
    with open(os.path.join(binPath, 'data.dat'), 'rb') as f:
        data = torch.tensor(pickle.load(f))
    print("No. stars: ", data[0].item())
    if data[0].item()==0:
        # No stars in bin
        with open(os.path.join(binPath, 'fit_results.dat'), 'wb') as f:
            pickle.dump([-999, -999, -999], f)
        return 0 # exits main(), skipping fit that would fail

    
    mu, D, R, modz, solidAngles, gLon, gLat, x, y, z = calc_coords(apo)
    effSelFunc = get_effSelFunc(binDict)
    multiplier = (solidAngles*(D**3)*(mu[1]-mu[0])*effSelFunc*np.log(10)/5)
    
    R_modz_multiplier = (R, modz, multiplier)
    
    print("bin: ", myUtils.binName(binDict))
    print('data: ', data)
    

    
    if MAP:
        guide = pyro.infer.autoguide.AutoDelta(model)
    else:
        guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    n_latents=3
    
    lr = 0.01
    optimiser = Adam({"lr": lr}) # tune, and look at other parameters
    svi = SVI(model, guide, optimiser, loss=Trace_ELBO(num_particles=4))

    print(f"Beginning fitting {datetime.datetime.now()}")
    maxSteps = 50000 # let settle significantly longer than visual loss decrease and median change stops - shape of guide is adjusting too
    lossArray = np.zeros(maxSteps)
    latent_medians = np.zeros((maxSteps,n_latents))
    incDetectLag = 1000
    for step in range(maxSteps):
        loss = svi.step(R_modz_multiplier, data)
        lossArray[step] = loss

        #parameter_means[step] = mvn_guide._loc_scale()[0].detach().numpy()
        latent_medians[step] = [v.item() for v in  guide.median(R_modz_multiplier).values()]
        
        
        # if loss>10**11: #step>incDetectLag and (lossArray[step]>lossArray[step-incDetectLag]):
        #     # Note: this is no longer the parts that checks convergence, that is below
        #     lossArray = lossArray[:step+1]
        #     latent_medians = latent_medians[:step+1]
        #     break
        
        if step%100==0:
            print(f'Loss = {loss}, median logNuSun = {latent_medians[step][0]}')
            
            if step>incDetectLag and np.all(np.abs(latent_medians[step]-latent_medians[step-incDetectLag])<np.array([0.01,0.01,0.01])):
                # tune condition if needed
                lossArray = lossArray[:step+1]
                latent_medians = latent_medians[:step+1]
                break
            
            
    print(f"finished fitting at step {step} {datetime.datetime.now()}")

    latent_names = list(guide.median(R_modz_multiplier).keys())
    print(latent_names)
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).data.cpu().numpy())

    for name, value in guide.median(R_modz_multiplier).items():
        print(name, value.item())

    logNuSun, loga_R, loga_z = guide.median(R_modz_multiplier).values()
    a_R = torch.exp(loga_R)
    a_z = torch.exp(loga_z)
    print("logNuSun, a_R, a_z:")
    print(logNuSun, a_R, a_z)


    with open(os.path.join(binPath, 'fit_results.dat'), 'wb') as f:
        print("What's saved:")
        print(logNuSun.item(), a_R.item(), a_z.item())
        pickle.dump([logNuSun.item(), a_R.item(), a_z.item()], f)


    distro = logNuSunDoubleExpPPP(logNuSun, a_R, a_z, *R_modz_multiplier)
    print(f"does {data[0]} equal {distro.effVol()}?")
    print(f"does {data[1]} equal {(distro.nu()*multiplier*R).sum()/distro.effVol()}?")
    print(f"does {data[2]} equal {(distro.nu()*multiplier*modz).sum()/distro.effVol()}?")
    print(f"does {data[0]*(data[1]-myUtils.R_Sun)} equal {(distro.nu()*multiplier*(R-myUtils.R_Sun)).sum()}?")
    print(f"does {data[0]*data[2]} equal {(distro.nu()*multiplier*modz).sum()}?")

    fig, ax = plt.subplots()
    ax.plot(lossArray)
    ax.set_xlabel("step number")
    ax.set_ylabel("loss")
    ax.set_title(f"lr: {lr}")
    fig.savefig(os.path.join(binPath, str(lr)+("-MAP-" if MAP else "-MVN-")+"loss.png")) #git hash?
    
    for i, name in enumerate(latent_names):
        fig, ax = plt.subplots()
        ax.plot(latent_medians[:,i])
        ax.set_xlabel("step number")
        ax.set_ylabel(name)
        ax.set_title(("MAP" if MAP else "MVN") +f", lr: {lr}")
        fig.savefig(os.path.join(binPath, str(lr)+("-MAP-" if MAP else "-MVN-")+name+".png")) #git hash?

    if not MAP:
        with pyro.plate("samples",5000,dim=-1):
            samples = guide(R_modz_multiplier) #samples latent space, accessed with samples["logA"] etc
            # as outputted as disctionary of tensors
        labels=latent_names
        samples_for_plot = np.stack([samples[label].detach().cpu().numpy() for label in labels],axis=-1)
        fig = corner.corner(samples_for_plot, labels=labels) #takes input as np array with rows as samples
        fig.suptitle(("MAP" if MAP else "MVN")+f", lr: {lr}, step: {step}")
        fig.savefig(os.path.join(binPath, str(lr)+("-MAP-" if MAP else "-MVN-")+"latents.png")) #git hash?

class logNuSunDoubleExpPPP(TorchDistribution): # move to own module?
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
        return torch.exp(self.logNuSun-self.a_R*(self.R - myUtils.R_Sun)-self.a_z*self.modz)

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
        return (value[0]*(self.logNuSun - (value[1]-myUtils.R_Sun)*self.a_R - value[2]*self.a_z)
                - self.effVol())

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Sample should not be required")


def model(R_modz_multiplier, data=None):
    """
    FeHBinEdges mark edges of metallicity bins being used, in form of 2*nbins tensor with [lowerEdge, upperEdge] for each bin
    sums is 3*nbins tensor with [N,sumR,summodz] for each bin
    FOR NOW ASSUMES ONE BIN
    
    20221101: changed priors to uniform in log. For logNuSun this is slightly arbitrary, but for scale length 
    and height this is the uninformative scale invariant prior
    """
    #with pyro.plate('bins', len(FeHBinEdges-1): # needs new bin definition: do something like with plate as
    #    logA = pyro.sample('logA', dist.Normal(...))
    #    a_R = pyro.sample('a_R', dist.Normal(...))
    #    a_z = pyro.sample('a_z', dist.Normal(...))
    #    return pyro.sample('obs', MyDist(FeHBinEdges, logA, a_R, a_z, validate_args=True), obs=sums)

    logNuSun = pyro.sample('logNuSun', distributions.Uniform(0, 20)) # tune these - check fitted values are nowhere near edge
    loga_R = pyro.sample('loga_R', distributions.Uniform(np.log(1/20), np.log(1/0.01)))
    loga_z = pyro.sample('loga_z', distributions.Uniform(np.log(1/20), np.log(1/0.01)))
    a_R = pyro.deterministic('a_R', torch.exp(loga_R))
    a_z = pyro.deterministic('a_z', torch.exp(loga_z))
    return pyro.sample('obs', logNuSunDoubleExpPPP(logNuSun, a_R, a_z, *R_modz_multiplier), obs=data)

def calc_coords(apo):
    """makes mu, D, R, modz, solidAngles, gLon gLat, and galacticentric
    x, y, and z arrays, for ease of use
    units are kpc, and R and modz is for central angle of field.
    rows are for fields, columns are for mu values"""
    locations = apo.list_fields(cohort='all')
    Nfields = len(locations)
    # locations is list of ids of fields with at least completed cohort of
    #  any type, therefore some stars in statistical sample
    mu = myUtils.arr(myUtils.muGridParams)
    D = 10**(-2+0.2*mu)
    gLon = np.zeros((Nfields, 1))
    gLat = np.zeros((Nfields, 1))
    solidAngles = np.zeros((Nfields, 1))
    # This shape allows clever broadcasting in coord.SkyCoord
    for loc_index, loc in enumerate(locations):
        gLon[loc_index,0], gLat[loc_index,0] = apo.glonGlat(loc)
        solidAngles[loc_index,0] = apo.area(loc)*(np.pi/180)**2 # converts deg^2 to steradians
    gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(myUtils.GC_frame)
    x = gCentricCoords.x.value
    y = gCentricCoords.y.value
    z = gCentricCoords.z.value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)
    return mu, D, R, modz, solidAngles, gLon, gLat, x, y, z

def get_effSelFunc(binDict):
    path = os.path.join(myUtils.clusterDataDir, 'bins', myUtils.binName(binDict), 'effSelFunc.dat')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            effSelFunc = pickle.load(f)
    else:
        raise FileNotFoundError("Currently effSelFunc must be calculated seperately")
    return effSelFunc


if __name__=='__main__':
    main()
