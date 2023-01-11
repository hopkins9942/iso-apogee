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


# def sampleiso(N, iso, weights='number', newgrid=False):
#     """
#     Sample isochrone recarray iso weighted by lognormal chabrier (2001) IMF (default in PARSEC)
#     """
#     weight_sort = np.argsort(iso['weights'])
#     inds = np.array(range(len(iso)))
#     inter = interp1d(np.cumsum(iso['weights'][weight_sort])/np.sum(iso['weights']), inds, kind='nearest')
#     random_indices = inter(np.random.rand(N)).astype(int)
#     return iso[weight_sort][random_indices]


if __name__=='__main__':
    # testing
    # Things to test: does things like my use of Mini look right, and can I get age distributions, and is IMF chabrier
    
    isogrid = newgrid()
    
    print('Mini: ', np.unique(isogrid['Mini']))
    print('ages: ', 10**(np.unique(isogrid['logAge']) - 9))
    
    youngMask = (10**(isogrid['logAge'] - 9) < 1)
    # print(youngMask)
    
    logAgeVals = np.unique(isogrid['logAge'])
    
    # print(np.unique(isogrid[youngMask]['Mini']))
    
    fig, ax = plt.subplots()
    ax.hist(isogrid[youngMask]['Mini'], weights=isogrid[youngMask]['weights'])
    ax.set_xlim(-1,6)
    ax.set_xlabel('Mini') # inital mass
    ax.set_yscale('log')
    
    fig, ax = plt.subplots()
    ax.hist(isogrid[youngMask]['Mass'], weights=isogrid[youngMask]['weights'])
    ax.set_xlim(-1,6)
    ax.set_xlabel('Mass') # actual mass including mass loss over time
    ax.set_yscale('log')
    
    MiniVals = np.unique(isogrid['Mini'])[0:-1:5000]
    # print(MiniVals)
    
    int_IMFVals = np.array([isogrid[np.where(isogrid['Mini']==mv)[0][0]]['int_IMF'] for mv in MiniVals])
    # print(int_IMFVals)
    
    MiniMids = (MiniVals[1:]+MiniVals[:-1])/2
    IMFVals = (int_IMFVals[1:]-int_IMFVals[:-1])/(MiniVals[1:]-MiniVals[:-1])
    
    fig, ax = plt.subplots()
    ax.plot(MiniVals, int_IMFVals)
    ax.set_xlabel('Mini')
    ax.set_ylabel('int_IMF') # integral of IMF up to Mini
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    
    fig, ax = plt.subplots()
    ax.plot(MiniMids, IMFVals)
    ax.plot(MiniMids, 10*0.158*(1/(MiniMids*np.log(10)))*np.exp(-0.5*((np.log10(MiniMids)-np.log10(0.22))/0.57)**2))
    ax.plot(MiniMids, 10*0.141*(1/(MiniMids*np.log(10)))*np.exp(-0.5*((np.log10(MiniMids)-np.log10(0.1))/0.627)**2))
    # Chab2001 was based on only 1 field and is what is used here, Chab 2003 is improved version
    ax.set_xlabel('Mini')
    ax.set_ylabel('IMF') # IMF at Mini
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    
    print(np.where(np.array([len(np.unique(isogrid[isogrid['Mini']==mv]['weights'])) for mv in MiniVals])!=1))
    # weight depends only on Mini, by IMF
    
    
    
    # Compare tracks
    masks = [(isogrid['MH']==0)&(isogrid['logAge']==av)&(isogrid['logg']>1)&(isogrid['logg']<=3)&(isogrid['Jmag']-isogrid['Ksmag']>0.3) for av in logAgeVals[1:]]
    # masks = [(isogrid['logAge']==av) for av in logAgeVals]
    
    fig, ax = plt.subplots()
    for m in masks:
        sortargs = np.argsort(isogrid[m]['Mini'])
        ax.plot(isogrid[m]['Jmag'][sortargs]-isogrid[m]['Ksmag'][sortargs], isogrid[m]['Hmag'][sortargs])
        print(isogrid[m&(isogrid['Hmag']>20)]['weights'].sum()/isogrid[m]['weights'].sum())
        print(isogrid[m]['weights'].sum())
        print()
    ax.set_ylabel('H')
    ax.invert_yaxis()
    ax.set_xlabel('J-Ks')
    #shoews a lot at H=30, white dwarfs? Do they affect my stats? This logg cut should only give giants
    #ESF puts this population at distance and calculates fraction observed
    # WDs will artifically reduce this population, albeit only has number (weight) fraction of 4e-4 at most
    # total weight decreases slightly with age, weird but only small
    
    
    # Age distribution - does RG area visibly change with reweighting, and how much does SineMorteMultiplier change
    
    RGareaMask = (isogrid['MH']==0)&(isogrid['logg']>1)&(isogrid['logg']<=3)&(isogrid['Jmag']-isogrid['Ksmag']>0.3)&(isogrid['logAge']>9)
    
    fig, ax = plt.subplots()
    ax.hist2d(isogrid[RGareaMask]['Jmag']-isogrid[RGareaMask]['Ksmag'], isogrid[RGareaMask]['Hmag'], weights=isogrid[RGareaMask]['weights'])
    ax.set_ylabel('H')
    ax.invert_yaxis()
    ax.set_xlabel('J-Ks')
    
    ywIsogrid = newgrid()
    ywIsogrid['weights'] = ywIsogrid['weights']*(14*10**9-10**(ywIsogrid['logAge']))
    
    
    owIsogrid = newgrid()
    owIsogrid['weights'] = owIsogrid['weights']*10**(ywIsogrid['logAge'])
    
    fig, ax = plt.subplots()
    ax.hist2d(ywIsogrid[RGareaMask]['Jmag']-ywIsogrid[RGareaMask]['Ksmag'], ywIsogrid[RGareaMask]['Hmag'], weights=ywIsogrid[RGareaMask]['weights'])
    ax.set_ylabel('H')
    ax.invert_yaxis()
    ax.set_xlabel('J-Ks')
    
    fig, ax = plt.subplots()
    ax.hist2d(owIsogrid[RGareaMask]['Jmag']-owIsogrid[RGareaMask]['Ksmag'], owIsogrid[RGareaMask]['Hmag'], weights=owIsogrid[RGareaMask]['weights'])
    ax.set_ylabel('H')
    ax.invert_yaxis()
    ax.set_xlabel('J-Ks')
    
    
    # fig, ax = plt.subplots()
    # ax.hist(np.unique(isogrid['int_IMF']))
    # ax.set_xlim(-1,6)
    
    
    # fig, ax = plt.subplots()
    # ax.hist(np.unique(isogrid['Mass']))
    # ax.set_xlim(-1,6)
    
    # fig, ax = plt.subplots()
    # ax.hist(np.unique(isogrid['Mini'][isogrid['logAge']<9]))
    # ax.set_xlim(-1,6)
    
    
    # fig, ax = plt.subplots()
    # ax.hist(np.unique(isogrid['int_IMF'][isogrid['logAge']<9]))
    # ax.set_xlim(-1,6)
    
    
    # fig, ax = plt.subplots()
    # ax.hist(np.unique(isogrid['Mass'][isogrid['logAge']<9]))
    # ax.set_xlim(-1,6)
    
    # fig, ax = plt.subplots()
    # ax.hist2d(10**(isogrid['logAge']), isogrid['Mini'])
    
    
    
    
    #'Zini','MH','logAge','Mini','int_IMF','Mass','logL','logTe','logg','label','McoreTP',
    #'C_O','period0','period1','period2','period3','period4','pmode','Mloss','tau1m',
    #'X','Y','Xc','Xn','Xo','Cexcess','Z','mbolmag','Jmag','Hmag','Ksmag','IRAC_3.6mag','IRAC_4.5mag','IRAC_5.8mag',
    #'IRAC_8.0mag','MIPS_24mag','MIPS_70mag','MIPS_160mag','W1mag','W2mag','W3mag','W4mag','weights'
    
    
    
    
    