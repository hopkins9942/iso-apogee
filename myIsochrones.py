import os

import numpy as np
from scipy import integrate
# from scipy.interpolate import interp1d
from astropy.io import ascii

from putStuffHere import dataDir

#just for testing
import matplotlib.pyplot as plt


def Kroupa(M):
    """
    Returns relative Kroupa number IMF at mass M
    0.84 calculated numerically to have integrated mass = 1 Sun
    additional factor to match int_IMF, doesn't affect results
    """
    weights = 0.84*np.where(M>=0.5, (M/0.5)**-2.3, (M/0.5)**-1.3)
    return weights*1.2

def Chab(M):
    weight = 0.141*(1/(M*np.log(10)))*np.exp(-0.5*((np.log10(M)-np.log10(0.1))/0.627)**2)
    return weight/0.0628 # integrated seperately, gives total mass =1 sun

# def addWeights(grid, IMF="Kroupa"):
#     """
#     Returns grid with added weights for Kroupa IMF
#     Not grid contains integrated IMF column, I use this to check
#     Don't use, append_fields is horribly memory efficient
#     """
#     if IMF=='Kroupa':
#         weights = Kroupa(grid['Mini'])
#     else:
#         raise NotImplemented("Add it yourself if you want it")
#     # return np.lib.recfunctions.append_fields(grid, 'weights', weights) 


def loadOldGrid():
    path = os.path.join(dataDir, 'input_data', 'PARSEC_lognormChab2001_linearagefeh.dat')
    table = ascii.read(path, guess=True)
    return table


def loadGrid():
    path = os.path.join(dataDir, 'input_data', 'PARSEC_Kroupa_isochrones.dat')
    return ascii.read(path, guess=True)
    

if __name__=='__main__':
    # testing
    # Things to test: does things like my use of Mini look right, and can I get age distributions, and is IMF chabrier
    # also definition of MH
    
    # Code here runs when when code run as script, use for testing
    
    isogrid = loadGrid()
    oldgrid = loadOldGrid()
    
    print('Mini: ', np.unique(isogrid['Mini']))
    print('log ages: ', np.unique(isogrid['logAge']))
    print('ages: ', 10**(np.unique(isogrid['logAge']) - 9))
    
    
    # Check individual isochrone
    isoMask = ((-0.0001<isogrid['MH'])&(isogrid['MH']<0.0251)
                 &(8.5>isogrid['logAge']))# gets one isochrone in both schemes
    iso = isogrid[isoMask]
    oldisoMask = ((-0.0001<oldgrid['MH'])&(oldgrid['MH']<0.0251)
                 &(8.5>oldgrid['logAge']))# gets one isochrone in both schemes
    oldiso = oldgrid[oldisoMask]
    
    
    print('MH: ', np.unique(iso['MH']))
    print('log ages: ', np.unique(iso['logAge'])) # checking only one isochrone
    
    # plotting single isochrone
    fig, ax = plt.subplots()
    ax.scatter(iso['Jmag']-iso['Ksmag'], iso['Hmag'],s=0.1)
    ax.set_ylabel('H')
    ax.invert_yaxis()
    ax.set_xlabel('J-Ks')
    ax.set_title('single isochrone, new')
    
    fig, ax = plt.subplots()
    ax.scatter(oldiso['Jmag']-oldiso['Ksmag'], oldiso['Hmag'],s=0.1)
    ax.set_ylabel('H')
    ax.invert_yaxis()
    ax.set_xlabel('J-Ks')
    ax.set_title('single isochrone, old')
    
    # Where do old grid weights come from:
    # Don't know where weights have come from, but they seem wrong
    fig, ax = plt.subplots()
    ax.plot((oldiso['Mini'][:-1]+oldiso['Mini'][1:])/2,
            (oldiso['int_IMF'][1:]-oldiso['int_IMF'][:-1])/(oldiso['Mini'][1:]-oldiso['Mini'][:-1]), '.')
    # approximate weight at midpoints using int_IMF, assumes Mini in order
    ax.plot(oldiso['Mini'], oldiso['weights'], '.')
    # given weights in file, incorrect
    ax.plot(oldiso['Mini'], Chab(oldiso['Mini']), '.')
    # actual weights calculated from IMF directly
    ax.set_xlabel('Mini')
    ax.set_ylabel('weights')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('Showing old weights are wrong')
    
    fig, ax = plt.subplots()
    ax.plot(oldiso['Mini'][1:],
            (oldiso['int_IMF'][1:]-oldiso['int_IMF'][:-1]), '.', alpha=0.5)
    # How i think weights were calculated, incorrectly
    ax.plot(oldiso['Mini'], oldiso['weights'], '.', alpha=0.5)
    # given weights in file, incorrect
    ax.set_xlabel('Mini')
    ax.set_ylabel('weights')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('Showing what old, incorrect weights are')
    
    # Checking new weighting is correct
    # by comparing to midpoint approximation from int_IMF
    fig, ax = plt.subplots()
    ax.plot((iso['Mini'][:-1]+iso['Mini'][1:])/2,
            (iso['int_IMF'][1:]-iso['int_IMF'][:-1])/(iso['Mini'][1:]-iso['Mini'][:-1]), '.') #assumes Mini in order
    
    ax.plot(iso['Mini'], Kroupa(iso['Mini']), '.')
    ax.set_xlabel('Mini')
    ax.set_ylabel('IMF')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('new')
    
    # and by plotting integrals
    fig, ax = plt.subplots()
    ax.plot(iso['Mini'], iso['int_IMF'])
    ax.plot(iso['Mini'], [integrate.quad(Kroupa, 0.05, m)[0]+0.5 for m in iso['Mini']])
    # arbitrary difference is arbitrary
    ax.set_xlabel('Mini')
    ax.set_ylabel('int_IMF') # integral of IMF up to Mini
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('new')
    
    
    
    
    # weights = np.zeros(len(isogrid))
    # weights[isogrid['Mini']>=0.5] = 0.2*isogrid['Mini'][isogrid['Mini']>=0.5]**-2.3
    # weights[isogrid['Mini']<0.5] = 2*0.2*isogrid['Mini'][isogrid['Mini']<0.5]**-1.3 #Kroupa canonical
    
    
    # print('Mini: ', np.unique(isogrid['Mini']))
    # print('ages: ', 10**(np.unique(isogrid['logAge']) - 9))
    
    # logAgeVals = np.unique(isogrid['logAge'])
    
    # MiniVals = np.unique(isogrid['Mini'])[0:-1:200]
    # # print(MiniVals)
    
    # int_IMFVals = np.array([isogrid[np.where(isogrid['Mini']==mv)[0][0]]['int_IMF'] for mv in MiniVals])
    # # print(int_IMFVals)
    
    # MiniMids = (MiniVals[1:]+MiniVals[:-1])/2
    # IMFVals = (int_IMFVals[1:]-int_IMFVals[:-1])/(MiniVals[1:]-MiniVals[:-1])
    
    # fig, ax = plt.subplots()
    # ax.plot(MiniVals, int_IMFVals)
    # ax.set_xlabel('Mini')
    # ax.set_ylabel('int_IMF') # integral of IMF up to Mini
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    
    
    
    
    
    # fig, ax = plt.subplots()
    # ax.plot(MiniMids, IMFVals)
    # ax.plot(MiniMids, [weights[np.where(isogrid['Mini']==m)[0][0]] for m in MiniMids])
    
    # # ax.plot(MiniMids, 10*0.158*(1/(MiniMids*np.log(10)))*np.exp(-0.5*((np.log10(MiniMids)-np.log10(0.22))/0.57)**2))
    # # ax.plot(MiniMids, 10*0.141*(1/(MiniMids*np.log(10)))*np.exp(-0.5*((np.log10(MiniMids)-np.log10(0.1))/0.627)**2))
    # # Chab2001 was based on only 1 field and is what is used here, Chab 2003 is improved version
    # ax.set_xlabel('Mini')
    # ax.set_ylabel('IMF') # IMF at Mini
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    
    # weights = np.zeros(len())
    
    
    # print(np.where(np.array([len(np.unique(isogrid[isogrid['Mini']==mv]['weights'])) for mv in MiniVals])!=1))
    # weight depends only on Mini, by IMF
    
    
    
    # Compare tracks
    # masks = [(isogrid['MH']==0)&(isogrid['logAge']==av)&(isogrid['logg']>1)&(isogrid['logg']<=3)&(isogrid['Jmag']-isogrid['Ksmag']>0.3) for av in logAgeVals[1:]]
    # masks = [(isogrid['logAge']==av) for av in logAgeVals]
    
    # fig, ax = plt.subplots()
    # for m in masks:
    #     sortargs = np.argsort(isogrid[m]['Mini'])
    #     ax.plot(isogrid[m]['Jmag'][sortargs]-isogrid[m]['Ksmag'][sortargs], isogrid[m]['Hmag'][sortargs])
    #     print(isogrid[m&(isogrid['Hmag']>20)]['weights'].sum()/isogrid[m]['weights'].sum())
    #     print(isogrid[m]['weights'].sum())
    #     print()
    # ax.set_ylabel('H')
    # ax.invert_yaxis()
    # ax.set_xlabel('J-Ks')
    #shoews a lot at H=30, white dwarfs? Do they affect my stats? This logg cut should only give giants
    #ESF puts this population at distance and calculates fraction observed
    # WDs will artifically reduce this population, albeit only has number (weight) fraction of 4e-4 at most
    # total weight decreases slightly with age, weird but only small
    
    
    # Age distribution - does RG area visibly change with reweighting, and how much does SineMorteMultiplier change
    
    # RGareaMask = (isogrid['MH']==0)&(isogrid['logg']>1)&(isogrid['logg']<=3)&(isogrid['Jmag']-isogrid['Ksmag']>0.3)&(isogrid['logAge']>9)
    
    # fig, ax = plt.subplots()
    # ax.hist2d(isogrid[RGareaMask]['Jmag']-isogrid[RGareaMask]['Ksmag'], isogrid[RGareaMask]['Hmag'], weights=isogrid[RGareaMask]['weights'])
    # ax.set_ylabel('H')
    # ax.invert_yaxis()
    # ax.set_xlabel('J-Ks')
    
    # ywIsogrid = newgrid()
    # ywIsogrid['weights'] = ywIsogrid['weights']*(14*10**9-10**(ywIsogrid['logAge']))
    
    
    # owIsogrid = newgrid()
    # owIsogrid['weights'] = owIsogrid['weights']*10**(ywIsogrid['logAge'])
    
    # fig, ax = plt.subplots()
    # ax.hist2d(ywIsogrid[RGareaMask]['Jmag']-ywIsogrid[RGareaMask]['Ksmag'], ywIsogrid[RGareaMask]['Hmag'], weights=ywIsogrid[RGareaMask]['weights'])
    # ax.set_ylabel('H')
    # ax.invert_yaxis()
    # ax.set_xlabel('J-Ks')
    
    # fig, ax = plt.subplots()
    # ax.hist2d(owIsogrid[RGareaMask]['Jmag']-owIsogrid[RGareaMask]['Ksmag'], owIsogrid[RGareaMask]['Hmag'], weights=owIsogrid[RGareaMask]['weights'])
    # ax.set_ylabel('H')
    # ax.invert_yaxis()
    # ax.set_xlabel('J-Ks')
    
    
    # Definition of MH:
    # randIndices = np.random.randint(len(isogrid), size=1000)
    # X = isogrid[randIndices]['X']
    # Y = isogrid[randIndices]['Y']
    # Z = isogrid[randIndices]['Z']
    # Zini = isogrid[randIndices]['Zini']
    # logAge = isogrid[randIndices]['logAge']
    # MH = isogrid[randIndices]['MH']
    
    
    
    # ageColours = np.zeros(len(logAge))-1
    # for i, la in enumerate(logAgeVals):
    #     ageColours[logAge==la] = i
    # print(np.where(ageColours==-1))
    
    
    # np.array([np.where(logAge==av)[0][0] for av in logAgeVals])
    
    # fig, ax = plt.subplots()
    # ax.scatter(Z,X+Y)
    # ax.set_ylabel('X+Y')
    # ax.set_xlabel('Z')
    # # X and Y are current, and vary with age
    
    # fig, ax = plt.subplots()
    # ax.scatter(Zini,Z, c=ageColours)
    # ax.set_ylabel('Z')
    # ax.set_xlabel('Zini')
    
    # fig, ax = plt.subplots()
    # ax.scatter(Zini, Y)
    # ax.set_ylabel('Y')
    # ax.set_xlabel('Zini')
    
    # def MH_f(Zini):
    #     return np.log10(Zini/(1-Zini-(0.2485+1.78*Zini))) - np.log10(0.0207)
    
    # fig, ax = plt.subplots()
    # ax.scatter(np.log10(Zini), MH) #STRAIGHT LINE! MH IS INITIAL
    # # ax.plot(np.log10(np.sort(Zini)), MH_f(np.sort(Zini)), c='C1')
    # # ax.hlines(0.7, -3,-1)
    # ax.set_ylabel('MH')
    # ax.set_xlabel('log10(Zini)')
    
    
    # fig, ax = plt.subplots()
    # ax.scatter(np.log10(Z), MH)
    # ax.set_ylabel('MH')
    # ax.set_xlabel('log10(Z)')
    
    
    
    
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
    
    
    
    
    