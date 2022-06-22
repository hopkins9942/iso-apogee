import DensityModelling_defs as dm

import sys

import numpy as np

import datetime
import pickle
import pyro
import pyro.distributions as distributions
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch

import matplotlib.pyplot as plt
import corner

BinNum = int(sys.argv[1])

edges = dm.arr((-1.025,0.475,0.1))
FeHBinEdges_array = [[edges[i],edges[i+1]] for i in range(len(edges)-1)]
print(FeHBinEdges_array)
with open(dm._ROOTDIR+'sav/data_20220618.dat', 'rb') as f:
    data_array=pickle.load(f)
print(data_array)


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


mvn_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
delta_guide = pyro.infer.autoguide.AutoDelta(model)

### main ###
print(f"Starting! {datetime.datetime.now()}")

#muMin = 4.0
#muMax = 17.0
#muDiff = 0.1
muGridParams = dm.muGridParams #(muMin, muMax, round((muMax-muMin)//muDiff))
apo = dm.load_apo()

mu, D, R, modz, solidAngles, gLon, gLat, x, y, z = dm.makeCoords(muGridParams, apo)

FeHBinEdges = FeHBinEdges_array[BinNum]
R_modz_multiplier = dm.calculate_R_modz_multiplier(FeHBinEdges, muGridParams)
*_, multiplier = R_modz_multiplier #R, modz comes from makeCoords

print(R.shape)
print(modz.shape)
print(multiplier.shape)
print(R.mean())
print(modz.mean())
print(multiplier.mean())

#data_array = [[],[],[],[10**5,8, 0.1],[],[]] #fill with values from _data - AS MEANS
#data = torch.tensor(data_array[BinNum])

#data_array = [[2.35200000e+03, 7.34630784e+00, 1.62761062e+00],
#              [1.45120000e+04, 9.24297959e+00, 1.01059230e+00],
#              [4.31350000e+04, 9.50945083e+00, 5.80609531e-01],
#              [5.48160000e+04, 8.87167417e+00, 3.69494077e-01],
#              [3.59720000e+04, 8.06852236e+00, 3.07138239e-01],
#              [6.55000000e+03, 6.88423645e+00, 3.26783708e-01]]

data = torch.tensor(data_array[BinNum,:])
print(f"Data: {data}")
print(f"BinNum: {BinNum}")

MAP = False
if MAP:
    guide = delta_guide
else:
    guide = mvn_guide
n_latents=3

lr = 0.05
optimiser = Adam({"lr": lr}) # tune, and look at other parameters
svi = SVI(model, guide, optimiser, loss=Trace_ELBO(num_particles=4))

print(f"Beginning fitting {datetime.datetime.now()}")
maxSteps = 10000 # let settle significantly longer than visual loss decrease and median change stops - shape of guide is adjusting too
lossArray = np.zeros(maxSteps)
latent_medians = np.zeros((maxSteps,n_latents))
for step in range(maxSteps):
    loss = svi.step(R_modz_multiplier, data)
    lossArray[step] = loss

    #parameter_means[step] = mvn_guide._loc_scale()[0].detach().numpy()
    latent_medians[step] = [v.item() for v in  guide.median(R_modz_multiplier).values()]
    incDetectLag = 200
    if loss>10**11: #step>incDetectLag and (lossArray[step]>lossArray[step-incDetectLag]):
        lossArray = lossArray[:step+1]
        latent_medians = latent_medians[:step+1]
        break

    if step%100==0:
        print(f'Loss = {loss}, median logNuSun = {latent_medians[step][0]}')
print(f"finished fitting at step {step} {datetime.datetime.now()}")

latent_names = list(guide.median(R_modz_multiplier).keys())
print(latent_names)
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())

for name, value in guide.median(R_modz_multiplier).items():
    print(name, value.item())

logNuSun, a_R, a_z = guide.median(R_modz_multiplier).values()
print(logNuSun, a_R, a_z)

picklePath = (dm._ROOTDIR + "sav/Results20220618/" +str(BinNum)
                    + ".dat")

with open(picklePath, 'wb') as f:
    pickle.dump([logNuSun, a_R, a_z], f)

distro = dm.logNuSunDoubleExpPPP(logNuSun, a_R, a_z, *R_modz_multiplier)
print(f"does {data[0]} equal {distro.effVol()}?")
print(f"does {data[1]} equal {(distro.nu()*multiplier*R).sum()/distro.effVol()}?")
print(f"does {data[2]} equal {(distro.nu()*multiplier*modz).sum()/distro.effVol()}?")

savePath = "/data/phys-galactic-isos/sjoh4701/APOGEE/outputs/DM_fit/"+str(BinNum)+"-"

fig, ax = plt.subplots()
ax.plot(lossArray)
ax.set_xlabel("step number")
ax.set_ylabel("loss")
ax.set_title(f"lr: {lr}")
fig.savefig(savePath+str(lr)+("-MAP-" if MAP else "-MVN-")+"loss.png")

for i, name in enumerate(latent_names):
    fig, ax = plt.subplots()
    ax.plot(latent_medians[:,i])
    ax.set_xlabel("step number")
    ax.set_ylabel(name)
    ax.set_title(("MAP" if MAP else "MVN") +f", lr: {lr}")
    fig.savefig(savePath+str(lr)+("-MAP-" if MAP else "-MVN-")+name+".png")

if not MAP:
    with pyro.plate("samples",5000,dim=-1):
        samples = guide(R_modz_multiplier) #samples latent space, accessed with samples["logA"] etc
        # as outputted as disctionary of tensors
    labels=latent_names
    samples_for_plot = np.stack([samples[label].detach().cpu().numpy() for label in labels],axis=-1)
    fig = corner.corner(samples_for_plot, labels=labels) #takes input as np array with rows as samples
    fig.suptitle(("MAP" if MAP else "MVN")+f", lr: {lr}, step: {step}")
    fig.savefig(savePath+str(lr)+("-MAP-" if MAP else "-MVN-")+"latents.png")

