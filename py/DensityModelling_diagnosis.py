import DensityModelling_defs as dm

import datetime

import numpy as np
import pyro
import pyro.distributions as distributions
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
#from matplotlib.colors import ListedColormap, BoundaryNorm
import corner


### defs ###

def model(R_modz_multiplier, sums=None):
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

    logA = pyro.sample('logA', distributions.Normal(18, 1)) # tune these
    a_R = pyro.sample('a_R', distributions.Normal(0.25, 0.09))
    a_z = pyro.sample('a_z', distributions.Normal(10, 1))
    return pyro.sample('obs', dm.oneBinDoubleExpPPP(logA, a_R, a_z, *R_modz_multiplier), obs=sums)

mvn_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
delta_guide = pyro.infer.autoguide.AutoDelta(model)

### main ###
print(f"Starting! {datetime.datetime.now()}")

FeHBinEdges = [-0.5,0]

muMin = 0.0
muMax = 12 #15 # CHECK THIS against allStar
# better to exclude fringe datapoints than have all data in nearest three
# bins - plot mu distribution
muDiff = 0.1
muGridParams = (muMin, muMax, int((muMax-muMin)//muDiff))
apo = dm.load_apo()

mu, D, R, modz, solidAngles, gLon, gLat, x, y, z = dm.makeCoords(muGridParams, apo)

R_modz_multiplier = dm.calculate_R_modz_multiplier(FeHBinEdges, muGridParams)
*_, multiplier = R_modz_multiplier #R, modz comes from makeCoords
mock_sums = (10**5)*torch.tensor([1,8,0.1])

MAP = False
if MAP:
    guide = delta_guide
else:
    guide = mvn_guide

n_latents = 3

lr = 0.005
optimiser = Adam({"lr": lr}) # tune, and look at other parameters
svi = SVI(model, guide, optimiser, loss=Trace_ELBO(num_particles=4))

print(f"Beginning fitting {datetime.datetime.now()}")
maxSteps = 20000 # let settle significantly longer than visual loss decrease and median change stops - shape of guide is adjusting too
lossArray = np.zeros(maxSteps)
latent_medians = np.zeros((maxSteps,n_latents))
for step in range(maxSteps):
    loss = svi.step(R_modz_multiplier, mock_sums)
    lossArray[step] = loss

    #parameter_means[step] = mvn_guide._loc_scale()[0].detach().numpy()
    latent_medians[step] = [v.item() for v in  guide.median(R_modz_multiplier).values()]
    incDetectLag = 200
    if loss>10**6: #step>incDetectLag and (lossArray[step]>lossArray[step-incDetectLag]):
        lossArray = lossArray[:step]
        latent_medians = latent_medians[:step]
        break

    if step%100==0:
        print(f'Loss = {loss}, median logA = {latent_medians[step][0]}')
print(f"finished fitting at step {step} {datetime.datetime.now()}")

#n_latents = len(guide.median(R_modz_multiplier))
#guide.median only works after guide instantiated in SVI
latent_names = list(guide.median(R_modz_multiplier).keys())
print(latent_names)
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())

for name, value in guide.median(R_modz_multiplier).items():
    print(name, value.item())


savePath = "/data/phys-galactic-isos/sjoh4701/APOGEE/outputs/"

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

medians_dict = guide.median(R_modz_multiplier)
a_R = medians_dict["a_R"].item() #(2*mock_sums[0]/mock_sums[1]).item()
a_z = medians_dict["a_z"].item() #(mock_sums[0]/mock_sums[2]).item()
B = dm.oneBinDoubleExpPPP(0, a_R, a_z, *R_modz_multiplier).effVol().item() #logA=0 makes effVol=B
logp = lambda logA: mock_sums[0].item()*(logA + 2*np.log(a_R) + np.log(a_z)) - mock_sums[1].item()*a_R - mock_sums[2].item()*a_z - B*np.exp(logA)
logAArray = np.linspace(16,24)

fig, ax = plt.subplots()
ax.set_xlabel("logA")
#ax.set_ylabel("-logp, loss")
ax.plot(logAArray,-logp(logAArray),'|',label="-logp") #could estimate posterior here
#sparselogAArray = np.linspace(20,24,5)
#ax.plot(sparselogAArray, [-oneBinDoubleExpPPP(lgA,a_R,a_z,*R_modz_multiplier).log_prob(mock_sums) for lgA in sparselogAArray])
#this was to test logp
ax.plot(latent_medians[:,0],lossArray,label="loss")
ax.legend()
fig.savefig(savePath+"logp.png")
print(f"B = {B}")

fig, ax = plt.subplots()
ax.set_xlabel("logA")
#ax.set_ylabel("effVol, N")
ax.plot(logAArray[::5], [dm.oneBinDoubleExpPPP(lgA,a_R,a_z,*R_modz_multiplier).effVol().item() for lgA in logAArray[::5]], label="effVol")
ax.plot(logAArray[::5], mock_sums[0]*np.ones(len(logAArray[::5])), label="N")
ax.legend()
fig.savefig(savePath+"effVol_N.png")


nu = dm.oneBinDoubleExpPPP(0,a_R,a_z,*R_modz_multiplier).norm_nu().detach().cpu().numpy() # I think cpu unneccessary for me, as I am not using GPU for now
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()
nu_flat = nu.flatten()
multiplier_flat = multiplier.flatten()
point_density = (5/np.log(10))*D**-3
point_density_flat = (point_density*np.ones_like(x)).flatten()
x_bins = np.linspace(-12,-4,20)
y_bins = np.linspace(-4,4,20)
z_bins = np.linspace(-1,1,5)

fig, ax = plt.subplots()
ax.set_xlabel("x /kpc")
ax.set_ylabel("y /kpc")
ax.set_aspect("equal")
ax.hist2d(x_flat, y_flat, bins=[x_bins,y_bins], weights=nu_flat/point_density_flat)
fig.colorbar(cm.ScalarMappable())
fig.savefig(savePath+"norm_nu_xy.png")

fig, ax = plt.subplots()
ax.set_xlabel("x /kpc")
ax.set_ylabel("z /kpc")
ax.set_aspect("equal")
ax.hist2d(x_flat, z_flat, bins=[x_bins,z_bins], weights=nu_flat/point_density_flat)
fig.colorbar(cm.ScalarMappable())
fig.savefig(savePath+"norm_nu_xz.png")

fig, ax = plt.subplots()
ax.set_xlabel("x /kpc")
ax.set_ylabel("y /kpc")
ax.set_aspect("equal")
ax.hist2d(x_flat, y_flat, bins=[x_bins,y_bins], weights=nu_flat*multiplier.flatten()/point_density_flat)
fig.colorbar(cm.ScalarMappable())
fig.savefig(savePath+"effVol_xy.png")

fig, ax = plt.subplots()
ax.set_xlabel("x /kpc")
ax.set_ylabel("z /kpc")
ax.set_aspect("equal")
ax.hist2d(x_flat, z_flat, bins=[x_bins,z_bins], weights=nu_flat*multiplier.flatten()/point_density_flat)
fig.colorbar(cm.ScalarMappable())
fig.savefig(savePath+"effVol_xz.png")


#Unfinished attempt at plotting each field as line in 3D:
#norm = plt.Normalize(nu_flat.min(), nu_flat.max())
#lc = LineCollection(segments, cmap='viridis', norm=norm)
## Set the values used for colormapping
#lc.set_array(dydx)
#lc.set_linewidth(2)
#line = axs[0].add_collection(lc)
#fig.colorbar(line, ax=axs[0])
#
#fig, ax = plt.subplots()
#ax.set_xlabel("x /kpc")
#ax.set_ylabel("z /kpc")
#ax.plot3d(x_flat, y_flat, z_flat, weights=nu_flat)
#fig.colorbar()
#fig.savefig(savePath+"norm_nu_xyz.png")
#
#fig, ax = plt.subplots()
#ax.set_xlabel("x /kpc")
#ax.set_ylabel("y /kpc")
#ax.plot3d(x_flat, y_flat, z_flat, weights=nu_flat*multiplier.flatten())
#fig.colorbar()
#fig.savefig(savePath+"effVol_xyz.png")




