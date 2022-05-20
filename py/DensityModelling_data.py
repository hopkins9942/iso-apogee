import DensityModelling_defs as dm

import apogee.tools.read as apread

import numpy as np
import pickle
import matplotlib.pyplot as plt
from astropy.coordinates import Galactocentric
import astropy.units as u


allStar = dm.load_allStar()
statIndx = dm.load_statIndx()

print(len(allStar))
print(np.count_nonzero(statIndx))

statSample = allStar[statIndx]

print(statSample.colnames)

fig, ax = plt.subplots()
ax.hist()
