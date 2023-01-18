import os

import numpy as np
from scipy import integrate
# from scipy.interpolate import interp1d
from astropy.io import ascii

from mySetup import dataDir

#just for testing
import matplotlib.pyplot as plt


minMini = 0.09
maxMini = 5.3535 
# taken from Kroupa isogrid

def Kroupa(M):
    """
    Returns relative Kroupa number IMF at mass M
    0.84 calculated numerically to have integrated mass = 1 Sun
    additional factor to match int_IMF, doesn't affect results.
    
    """
    weights = 0.84*np.where(M>=0.5, (M/0.5)**-2.3, (M/0.5)**-1.3)
    return weights*1.23

minMini = 0.09
maxMini = 5.3535 
# taken from Kroupa isogrid, used for integrals
weightPerIsochrone = integrate.quad(Kroupa, minMini, maxMini)[0]
meanMini = integrate.quad(lambda m:m*Kroupa(m), minMini, maxMini)[0]/weightPerIsochrone


def Chab(M):
    weight = 0.141*(1/(M*np.log(10)))*np.exp(-0.5*((np.log10(M)-np.log10(0.1))/0.627)**2)
    # 
    return 0.95*weight/0.0628 # integrated seperately, gives total mass =1 sun


def extractIsochrones(isogrid):
    """finds values of MH and age at which isochrones calculated,
    and frst index of each"""
    MH_logAge = np.column_stack((isogrid['MH'], isogrid['logAge']))
    return np.unique(MH_logAge, return_index=True, axis=0)
    
    
def calcWeights(isogrid):
    """
    Weight of each point is proportional to number of stars on isochrone
    between previous mass point and this one. for first mass point, equal to Kroupa
    integrated from lowest Mini in isogrid
    """
    MH_logAge_vals, indices  = extractIsochrones(isogrid)
    diff = np.zeros(len(isogrid))
    diff[1:] = isogrid['int_IMF'][1:]-isogrid['int_IMF'][:-1]
    diff[indices] = np.array([integrate.quad(Kroupa, minMini, m)[0] for m in isogrid['Mini'][indices]])
    return diff


def loadOldGrid():
    path = os.path.join(dataDir, 'input_data', 'PARSEC_lognormChab2001_linearagefeh.dat')
    table = ascii.read(path, guess=True)
    return table


def loadGrid():
    path = os.path.join(dataDir, 'input_data', 'PARSEC_Kroupa_isochrones.dat')
    return ascii.read(path, guess=True)


def makeRGmask(isogrid):
    return ((1<=isogrid['logg']) & (isogrid['logg']<3) & (isogrid['Jmag']-isogrid['Ksmag']>0.3))


def NRG2SMM(isogrid):
    """
    Returns ratio between the  sine morte mass and red giant number in isogrid
    isogrid can be swapped for a single or multiple whole isochrones, as long as
    they are whole and ordered correctly
    """
    weights = calcWeights(isogrid)
    RGmask = makeRGmask(isogrid)
    weightinRG = weights[RGmask].sum()
    
    return meanMini*weightPerIsochrone/weightinRG
    

def test():
    # testing
    # Things to test: does things like my use of Mini look right, and can I get age distributions, and is IMF chabrier
    # also definition of MH
    
    # Code here runs when when code run as script, use for testing
    
    isogrid = loadGrid()
    oldgrid = loadOldGrid()
    
    print('Mini: ', np.unique(isogrid['Mini']))
    print('log ages: ', np.unique(isogrid['logAge']))
    print('ages: ', 10**(np.unique(isogrid['logAge']) - 9))
    
    # Where do grid weights come from: diff int_IMF
    
    oldisoMask = ((-0.0001<oldgrid['MH'])&(oldgrid['MH']<0.0251)
                 &(8.5>oldgrid['logAge']))
    oldiso = oldgrid[oldisoMask]
    
    fig, ax = plt.subplots()
    ax.plot(oldiso['Mini'],
            np.concatenate(([oldiso['int_IMF'][0]], oldiso['int_IMF'][1:]-oldiso['int_IMF'][:-1])),
            '.', alpha=0.5)
    ax.plot(oldiso['Mini'], oldiso['weights'], '.', alpha=0.5)
    ax.plot(oldiso['Mini'], calcWeights(oldgrid)[oldisoMask], '.', alpha=0.5)
    ax.set_xlabel('Mini')
    ax.set_ylabel('weights')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('Showing new scheme matches old')
    
    
    # individiual isochrones:
    isogrid = loadGrid()
    MH_logAge, indx = extractIsochrones(isogrid)
    weights = calcWeights(isogrid)
    
    saveDir = os.path.join(dataDir,'outputs','myIsochrones')
    
    for i in np.random.randint(0, len(indx), 10): #pick random isochrones
        MH, logAge = MH_logAge[i]
        if i<len(indx)-1:
            isoIndices = np.arange(indx[i], indx[i+1])
        else:
            isoIndices = np.arange(indx[i],len(isogrid))
        iso = isogrid[isoIndices]
        
        print(f'NRG2SMM: {NRG2SMM(iso)}, age={10**(MH_logAge[i,1]-9)}')
        
        # plot Isochrone
        fig, ax = plt.subplots()
        ax.plot(iso['Jmag']-iso['Ksmag'], iso['Hmag'])
        ax.set_ylabel('H')
        ax.invert_yaxis()
        ax.set_xlabel('J-Ks')
        ax.set_title(f'MH={MH}, logAge={logAge}')
        # path = os.path.join(saveDir,f'iso_{MH}_{logAge}.png')
        # fig.savefig(path, dpi=200)
        
        
        # plot IMF
        bins = np.logspace(np.log10(iso['Mini'].min()/1.001), np.log10(iso['Mini'].max()*1.001), 10)
        binWidths = bins[1:]-bins[:-1]
        
        # fracerror = (np.histogram(iso['Mini'], bins)[0])**-0.5
        # hist = np.histogram(iso['Mini'], bins, weights = weights[isoIndices]/(binWidths[np.digitize(iso['Mini'],bins)-1]))[0]
        
        fig, ax = plt.subplots()
        ax.hist(iso['Mini'], weights=weights[isoIndices]/(binWidths[np.digitize(iso['Mini'],bins)-1]),
        bins=bins)
        ax.plot(iso['Mini'], Kroupa(iso['Mini']))
        ax.plot(iso['Mini'], np.zeros_like(iso['Mini']),  '.')
        # ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('IMF')
        ax.set_xlabel('Mini')
        ax.set_title(f'MH={MH}, logAge={logAge}')
        # Note: first bar here almost always underestimates as is sum only of weights of points in bin
        # Meaning vlaue of bin is achieved by integrating Kroupa part-way across, then dividing by full width of bin
        # graph below shows it works
        # path = os.path.join(saveDir,f'IMF_{MH}_{logAge}.png')
        # fig.savefig(path, dpi=200)
        
        # Better way
        fig, ax = plt.subplots()
        ax.plot(iso['Mini'], np.cumsum(weights[isoIndices]))
        ax.plot(iso['Mini'], [integrate.quad(Kroupa, minMini, m)[0] for m in iso['Mini']], alpha=0.5)
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.set_ylabel('int_IMF')
        ax.set_xlabel('Mini')
        ax.set_title(f'MH={MH}, logAge={logAge}')
        # path = os.path.join(saveDir,f'cumIMF_{MH}_{logAge}.png')
        # fig.savefig(path, dpi=200)
    
    
    # Test Red giantsample looks right:
    RGmask = ((1<=isogrid['logg']) & (isogrid['logg']<3) & (isogrid['Jmag']-isogrid['Ksmag']>0.3))
    fig, ax = plt.subplots()
    ax.scatter(isogrid[RGmask]['Jmag']-isogrid[RGmask]['Ksmag'], isogrid[RGmask]['Hmag'], s=0.1)
    ax.set_ylabel('H')
    ax.invert_yaxis()
    ax.set_xlabel('J-Ks')
    ax.set_title('Red giants')
    #NB: colour cut in mask cuts white dwarfs and any stars which would not be in any apogee fields
    # as every field has either (J-Ks)0>0.3 or 0.5
    # NB: upper segment of this distribution is entirely youngest population of stars (therefore most massive) 
    # This may make ESFs quite age dependent - something I should test
    
    
        
    
if __name__=='__main__':
    test()
    pass
    
    