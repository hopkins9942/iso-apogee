import DensityModelling_defs as dm

import apogee.tools.read as apread

import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u

#Plot distributions in colour, magnitude, metallicity, alpha and age
# Compare obs distribution to distributions of effVol
# Compare my own distributions - does hole in xz match up to gap in b for mod(l)<90?
# do a test fit with logNuSun to check it works
# Work out how to deal with cuts and bad stars - think about total mass - then fit real data!

allStar = dm.load_allStar()
statIndx = dm.load_statIndx()

print(len(allStar))
print(np.count_nonzero(statIndx))

statSample = allStar[statIndx]

# print(statSample.dtype.names)

def extract(S):
    gLon = S['GLON']
    gLat = S['GLAT']
    D = S['weighted_dist']/1000 # kpc
    mu = 10 + 5*np.log10(D)
    gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(dm.GC_frame)
    x = gCentricCoords.x.value
    y = gCentricCoords.y.value
    z = gCentricCoords.z.value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)

    FeH = S['M_H'] #Check with Ted
    alphaFe = S['ALPHA_M'] #check
    age = S['age_lowess_correct'] #check

    return mu, D, R, modz, gLon, gLat, x, y, z, FeH, alphaFe, age
mu, D, R, modz, gLon, gLat, x, y, z, FeH, alphaFe, age = extract(statSample)

savePath = "/data/phys-galactic-isos/sjoh4701/APOGEE/outputs/DM_data/"


fig, ax = plt.subplots()
ax.hist2d([l if l<180 else l-360 for l in gLon],gLat, bins=[100,50])
ax.set_xlabel("l/deg")
ax.set_ylabel("b/deg")
fig.savefig(savePath+"glonglat.png")

x_bins = np.linspace(-20,5,100)
y_bins = np.linspace(-10,10,80)
z_bins = np.linspace(-5,5,40)
D_bins = np.linspace(0,20,80)
logD_bins = np.logspace(np.log10(0.0001), np.log10(20),80)
mu_bins = np.linspace(-5,25,80)
R_bins = np.linspace(0,20,80)
logR_bins = np.logspace(np.log10(0.1), np.log10(20),80)
FeH_bins = np.linspace(-2,1,50)
alphaFe_bins = np.linspace(-0.2,0.4,50)
age_bins = np.linspace(-1,14,50)

fig, ax = plt.subplots()
ax.hist2d(x,y,bins=[x_bins,y_bins],norm=mpl.colors.LogNorm())
ax.set_aspect("equal")
ax.set_xlabel("x/kpc")
ax.set_ylabel("y/kpc")
ax.set_title("log(number in statistical sample)")
fig.savefig(savePath+"xy.png")

fig, ax = plt.subplots()
ax.hist2d(x,z,bins=[x_bins,z_bins],norm=mpl.colors.LogNorm())
ax.set_aspect("equal")
ax.set_xlabel("x/kpc")
ax.set_ylabel("z/kpc")
ax.set_title("log(number in statistical sample)")
fig.savefig(savePath+"xz.png")
# do as corner plot?

fig, ax = plt.subplots()
ax.hist(D, bins=D_bins)
ax.set_xlabel("D/kpc")
fig.savefig(savePath+"D.png")

fig, ax = plt.subplots()
ax.hist(D, bins=logD_bins)
ax.set_xlabel("D/kpc")
ax.set_xscale('log')
fig.savefig(savePath+"logD.png")

fig, ax = plt.subplots()
ax.hist(mu, bins=mu_bins)
ax.set_xlabel("mu")
fig.savefig(savePath+"mu.png")

fig, ax = plt.subplots()
ax.hist(R, bins=R_bins)
ax.set_xlabel("R/kpc")
fig.savefig(savePath+"R.png")

fig, ax = plt.subplots()
ax.hist(R, bins=logR_bins)
ax.set_xlabel("R/kpc")
ax.set_xscale('log')
fig.savefig(savePath+"logR.png")

fig, ax = plt.subplots()
ax.hist(z, bins=z_bins)
ax.set_xlabel("z/kpc")
fig.savefig(savePath+"z.png")


fig, ax = plt.subplots()
ax.hist(FeH, bins=FeH_bins)
ax.set_xlabel("[Fe/H]")
fig.savefig(savePath+"FeH.png")

fig, ax = plt.subplots()
ax.hist(alphaFe, bins=alphaFe_bins)
ax.set_xlabel(r"[$\alpha/$Fe]")
fig.savefig(savePath+"alphaFe.png")

fig, ax = plt.subplots()
ax.hist2d(FeH,alphaFe,bins=[FeH_bins,alphaFe_bins]) # ,norm=mpl.colors.LogNorm())
ax.set_aspect("equal")
ax.set_xlabel("[Fe/H]")
ax.set_ylabel(r"[$\alpha /$Fe]")
fig.savefig(savePath+"alphaFeFeH.png")

fig, ax = plt.subplots()
ax.hist(age, bins=age_bins)
ax.set_xlabel("age/Gyr")
fig.savefig(savePath+"age.png")


# Cuts:

FeHBinEdges_array = [[-1.0,-0.75], [-0.75,-0.5], [-0.5,-0.25], [-0.25,0.0], [0.0,0.25], [0.25,0.5]]
sums_array = np.zeros([len(FeHBinEdges_array), 3])
for i in range(len(FeHBinEdges_array)):
    indices = (FeHBinEdges_array[i][0]<=FeH)and(FeH<FeHBinEdges_array[i][1])and(4.0<=mu)and(mu<17)
    sums_array[i][0] = np.count_nonzeros(indices)
    sums_array[i][1]
