# Goal is unified, easy plotting of any combination of bins

import numpy as np
import os
import pickle
import git

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

import myUtils


cmap1 = mpl.colormaps['Blues']
cmap2 = mpl.colormaps['hsv']