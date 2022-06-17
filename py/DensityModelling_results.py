import DensityModelling_defs as dm

import numpy as np
import matplotlib.pyplot as plt

FeHBinEdges = np.array(dm.FeHBinEdges_array)

savePath = "/data/phys-galactic-isos/sjoh4701/APOGEE/outputs/DM_results/"

results = np.array(dm.results)
# adjust for bad stars here
nuSun = np.log(results[:,0])
hR = 1/results[:,1]
hz = 1/results[:,2]

left_edges = FeHBinEdges[:,0]

fig, ax = plt.subplots()
ax.bar(left_edges,nuSun,width=0.25, align='edge')
ax.set_xlabel('[Fe/H]')
ax.set_ylabel('number density of red giants at solar radius/kpc^-3')
fig.savefig(savePath + 'nuSun.png')

fig, ax = plt.subplots()
ax.bar(left_edges,hR,width=0.25, align='edge')
ax.set_xlabel('[Fe/H]')
ax.set_ylabel('scale length/kpc')
fig.savefig(savePath + 'hR.png')

fig, ax = plt.subplots()
ax.bar(left_edges,hz,width=0.25, align='edge')
ax.set_xlabel('[Fe/H]')
ax.set_ylabel('scale hight/kpc')
fig.savefig(savePath + 'hz.png')

