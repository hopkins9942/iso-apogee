import os

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from astropy.io import ascii

from myUtils import dataDir


#just for testing
import matplotlib.pyplot as plt




def newgrid():
    path = os.path.join(dataDir, 'input_data', 'PARSEC_lognormChab2001_linearagefeh.dat')
    table = ascii.read(path, guess=True)
    return table


def sampleiso(N, iso, weights='number', newgrid=False):
    """
    Sample isochrone recarray iso weighted by lognormal chabrier (2001) IMF (default in PARSEC)
    """
    weight_sort = np.argsort(iso['weights'])
    inds = np.array(range(len(iso)))
    inter = interp1d(np.cumsum(iso['weights'][weight_sort])/np.sum(iso['weights']), inds, kind='nearest')
    random_indices = inter(np.random.rand(N)).astype(int)
    return iso[weight_sort][random_indices]


if __name__=='__main__':
    # testing
    
    isogrid = newgrid()
    
    fig, ax = plt.subplots()
    ax.hist(np.unique(isogrid['Mini']))
    ax.set_xlim(-1,6)
    
    
    fig, ax = plt.subplots()
    ax.hist(np.unique(isogrid['int_IMF']))
    ax.set_xlim(-1,6)
    
    
    fig, ax = plt.subplots()
    ax.hist(np.unique(isogrid['Mass']))
    ax.set_xlim(-1,6)
    
    fig, ax = plt.subplots()
    ax.hist(np.unique(isogrid['Mini'][isogrid['logAge']<9]))
    ax.set_xlim(-1,6)
    
    
    fig, ax = plt.subplots()
    ax.hist(np.unique(isogrid['int_IMF'][isogrid['logAge']<9]))
    ax.set_xlim(-1,6)
    
    
    fig, ax = plt.subplots()
    ax.hist(np.unique(isogrid['Mass'][isogrid['logAge']<9]))
    ax.set_xlim(-1,6)
    
    fig, ax = plt.subplots()
    ax.hist2d(10**(isogrid['logAge']), isogrid['Mini'])
    
    
    
    
    #'Zini','MH','logAge','Mini','int_IMF','Mass','logL','logTe','logg','label','McoreTP',
    #'C_O','period0','period1','period2','period3','period4','pmode','Mloss','tau1m',
    #'X','Y','Xc','Xn','Xo','Cexcess','Z','mbolmag','Jmag','Hmag','Ksmag','IRAC_3.6mag','IRAC_4.5mag','IRAC_5.8mag',
    #'IRAC_8.0mag','MIPS_24mag','MIPS_70mag','MIPS_160mag','W1mag','W2mag','W3mag','W4mag','weights'