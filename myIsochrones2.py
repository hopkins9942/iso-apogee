import os

import numpy as np
from scipy import optimize
# from scipy.interpolate import interp1d
from astropy.io import ascii

from mySetup import dataDir

#just for testing
import matplotlib.pyplot as plt
 
BFk = 1.03083285

normk = 0.62273456372



def Kroupa(M, k=normk):
    """
    Returns relative Kroupa number IMF at mass M.
    
    Variously called the universal, standard, canonical or average IMF, and is 
    supposedly corrected for multiple systems accouding to Kroupa et al. 2013.
    Has a minium mass of 0.08 or 0.07 and a maximum of 150, variously in
    https://ui.adsabs.harvard.edu/abs/2013pss5.book..115K/abstract
    https://ui.adsabs.harvard.edu/abs/2002ASPC..285...86K/abstract
    https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K/abstract
    
    I am settling on a mass range of 0.08-150, and an amplitude which puts total
    NUMBER in this range to 1.
    A difference in int_IMF in the isochrone table can be handily converted
    to my normalisation by multipying by normk/BFk. To get int_IMF to my value,
    also need to subtract offset, equal to intKroupa(0.08,minM=0.031)
    """
    val = np.where(M<0.5, (M/0.5)**-1.3, (M/0.5)**-2.3)
    return k*val

def intKroupa(M, k=normk, minM=0.08):
    """Integral of Kroupa IMF, seeing as it's easily done analytically
    assumes minimum mass of 0.08 Msun or 0.07 and ignores possibility of maximum mass
    """    
    integral = np.where(M<minM, 0, 
                    np.where(M<0.5, (1/0.6)*((minM/0.5)**-0.3 - (M/0.5)**-0.3),
                             (1/0.6)*((minM/0.5)**-0.3 - 1) + (1/2.6)*(1-(M/0.5)**-1.3)))
    return k*integral

def MintKroupa(M, k=normk, minM=0.08):
    """integral of mass times Kroupa
   Note minM mustbe <0.5
   To get average sine morte mass, do MintKroupa(150)
   as already number normalised :)"""
    integral= np.where(M<minM, 0, 
                    np.where(M<0.5, (1/2.8)*((M/0.5)**0.7-(minM/0.5)**0.7),
                             (1/2.8)*(1-(minM/0.5)**0.7) + (1/1.2)*(1-(M/0.5)**-0.3)))
    return k*integral

def noUpperLimTotalMassKroupa(k=normk, minM=0.08):
    return (k/2.8)*(1-(minM/0.5)**0.7) + (k/1.2)



rBH = 1-intKroupa(25)

meanMini = MintKroupa(150)    



# def KroupaBD(M):
#     """
#     Returns relative Kroupa number IMF at mass M.
#     Includes Brown dwarfs as used in Sajadian 23
#     """
#     minBDM = 0.01
#     weights = np.where(M>=0.5, (M/0.5)**-2.3, np.where(M<0.08, ((0.08/0.5)**-1.3)*((M/0.08)**-0.7), (M/0.5)**-1.3))
#     return weights*1

def makeRGmask(isogrid):
    return ((1<=isogrid['logg']) & (isogrid['logg']<3) & (isogrid['Jmag']-isogrid['Ksmag']>0.3))

def NRG2SMM(isogrid, tau0, omega):
    """
    Returns ratio between the  sine morte mass and red giant number in isogrid
    isogrid can be swapped for a single or multiple whole isochrones, as long as
    they are whole and ordered correctly
    
    ageWeighting uses same scheme as ESF - weight calculated per isochrone not per
    point as there are more points on some isochrones
    """
    
    MH_logAge, indices = extractIsochrones(isogrid)
    
    ages = 10**(MH_logAge[:,1]-9)
    ageWeighting = np.exp(-(omega/2)*(ages-tau0)**2)
    ageWeighting/=ageWeighting.sum()
    # gives each isochrone a weight, such that sum of weights is 1
    
    weights = calcWeights(isogrid)*ageWeighting[np.digitize(np.arange(len(isogrid)), indices)-1]
    # multiplicatively increases isopoint weight by the age weighting for that point's isochrone
    # digitize takes arange 0,...,len(isogrid)-1, then for each works out the index of the isochrone each isopoint is on.
    

    RGmask = makeRGmask(isogrid)
    weightinRG = weights[RGmask].sum()
    # Since ageWeighting is normalised, weightinRG is the weighted mean of the IMF-weight  
    # in each isochrone in the RG region
    
    # Since total SM weight of each isochrone is 1, number fraction in RG is weightinRG
    return meanMini/weightinRG

def NRG2NLS(isogrid, tau0, omega):
    """
    Returns ratio between the  number of living stars and red giant number in isogrid
    isogrid can be swapped for a single or multiple whole isochrones, as long as
    they are whole and ordered correctly
    """
    
    MH_logAge, indices = extractIsochrones(isogrid)
    
    ages = 10**(MH_logAge[:,1]-9)
    ageWeighting  = np.exp(-(omega/2)*(ages-tau0)**2)
    ageWeighting/=ageWeighting.sum()
    # gives each isochrone a weight, such that sum of weights is 1
    
    weights = calcWeights(isogrid)*ageWeighting[np.digitize(np.arange(len(isogrid)), indices)-1]
    # multiplicatively increases isopoint weight by the age weighting for that point's isochrone
    # digitize takes arange 0,...,len(isogrid)-1, then for each works out index of isochrone .
    

    RGmask = makeRGmask(isogrid)
    weightinRG = weights[RGmask].sum()
    # Since ageWeighting is normalised, weightinRG is the weighted mean of the IMF-weight  
    # in each isochrone in the RG region
    
    
    # Living stars equal to sum of weight in array
    # as isochrones automatically ignore high-mass, dead points
    return weights.sum()/weightinRG
    

def NRG2NSM(isogrid, tau0, omega):
    """
    Returns ratio between the  number of sine morte stars and red giant number in isogrid
    Essentially NRG2SM without multiplication by mean mass
    isogrid can be swapped for a single or multiple whole isochrones, as long as
    they are whole and ordered correctly
    """
    
    MH_logAge, indices = extractIsochrones(isogrid)
    
    ages = 10**(MH_logAge[:,1]-9)
    ageWeighting  = np.exp(-(omega/2)*(ages-tau0)**2)
    ageWeighting/=ageWeighting.sum()
    # gives each isochrone a weight, such that sum of weights is 1
    
    weights = calcWeights(isogrid)*ageWeighting[np.digitize(np.arange(len(isogrid)), indices)-1]
    # multiplicatively increases isopoint weight by the age weighting for that point's isochrone
    # digitize takes arange 0,...,len(isogrid)-1, then for each works out index of isochrone .
    

    RGmask = makeRGmask(isogrid)
    weightinRG = weights[RGmask].sum()
    # Since ageWeighting is normalised, weightinRG is the weighted mean of the IMF-weight  
    # in each isochrone in the RG region
    
    # sine morte distribution requires weightPerIsochrone,
    # not sum of weights as old isochrones lack high-mass, dead points
    return 1/weightinRG










def testIMF(mh=0.05,age=0.5e9):
    """Demonstrates IMF used by isochrone int_IMF is equal to what I have"""
    isogrid = loadGrid()
    # isochrone = isogrid[(0<=isogrid['MH'])&(isogrid['MH']<0.1)&(isogrid['logAge']<np.log10(1e9))] this works, but maybe more pythonic is
    MH_logAge, indx = extractIsochrones(isogrid)

    iso_num = np.nonzero(np.isclose(MH_logAge, (mh,np.log10(age))).all(axis=1))[0][0]
    # isochrone = isogrid[indx[iso_num]:(indx[iso_num+1] if (iso_num+1!=len(indx)) else -1)] #old way
    indx = np.append(indx, len(isogrid))
    isochrone = isogrid[indx[iso_num]:indx[iso_num+1]] #new way
    
    int_IMFdiffs = isochrone['int_IMF'][1:] - isochrone['int_IMF'][:-1]
    Mmids = (isochrone['Mini'][1:] + isochrone['Mini'][:-1])/2
    Mdiffs = isochrone['Mini'][1:] - isochrone['Mini'][:-1]
    
    fig,ax = plt.subplots()
    ax.plot(Mmids, int_IMFdiffs/Mdiffs, label='Isochrone')
    ax.plot(Mmids, Kroupa(Mmids, k=BFk), linestyle=':', label='Function')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Initial Mass / $M_\odot$')
    ax.set_ylabel(r'IMF / $M_\odot^{-1}$')
    ax.legend()
    # Demonstrates that int_IMF is same as my Kroupa function
    # In isochrone file: Kroupa (2001, 2002) + Kroupa et al. (2013) canonical two-part-power law IMF corrected for unresolved binaries 
    # papers that make it most obvious is
    # https://ui.adsabs.harvard.edu/abs/2013pss5.book..115K/abstract
    # https://ui.adsabs.harvard.edu/abs/2002ASPC..285...86K/abstract
    # https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K/abstract
    # Weirdly, the 2013 paper canonical/standard/average IMF is corrected for binaries,
    # but this form is equal to the uncorrected univeral and aveage forms in the 2001 and 2002 papers.
    # Also the supposed corrected forms in to 2001 and 2002 papers are different.
    
    # K2013 has a different minimum mass (0.07 vs 0.08) -  now investigating which used by PARSEC and what the first int_IMF on the isochrone comes from
    fig,ax = plt.subplots()
    ax.plot(isochrone['Mini'], isochrone['int_IMF'], label='Isochrone')
    ax.plot(isochrone['Mini'], intKroupa(isochrone['Mini'],k=BFk,minM=0.08), linestyle='-.', label='minM=0.08')
    ax.plot(isochrone['Mini'], intKroupa(isochrone['Mini'],k=BFk,minM=0.031), linestyle='-.', label='minM=0.031')
    ax.set_xscale('log')
    ax.set_xlabel(r'Initial Mass / $M_\odot$')
    ax.set_ylabel(r'Cumulative IMF')
    ax.legend()
    # seems int_IMF of lowest mass star (always 0.09, sim to literature for lowest mass stars)
    # is calculated with minimum of 0.031 - far too low
    # very starnge value, and I shouldn't use it. Don't need to understand it's origin, but it could be 
    # they integrate up from Kroupa's extension to brown dwarfs
        
    #best way to get exact PARSEC values is camparing int_IMF
    def fun(x):
        k, minM = x
        residualSq = (intKroupa(isochrone['Mini'],k,minM)
                      - isochrone['int_IMF'])**2
        return np.sum(residualSq)
    res = optimize.minimize(fun, (1.031,0.031))
    print(res)
    print(f'Best fit to PARSEC k = {res.x[0]:.8f}, minM = {res.x[1]:.8f}')
    
    #finally testing int_IMF normalisation is such that total mass=1
    fig,ax = plt.subplots()
    ax.plot(Mmids, np.cumsum(Mmids*int_IMFdiffs), label='Isochrone')
    # ax.plot(isochrone['Mini'][:-1], np.cumsum(isochrone['Mini'][:-1]*int_IMFdiffs))
    # ax.plot(isochrone['Mini'][1:], np.cumsum(isochrone['Mini'][1:]*int_IMFdiffs))
    ax.plot(Mmids, MintKroupa(Mmids,k=BFk,minM=0.08), label='minM=0.08')
    ax.plot(Mmids, MintKroupa(Mmids,k=BFk,minM=0.07), label='minM=0.07')
    ax.plot(Mmids, MintKroupa(Mmids,k=BFk,minM=0.031), label='minM=0.031')
    ax.plot(Mmids, MintKroupa(Mmids,k=BFk,minM=isochrone['Mini'].min()), label='minM=min on isochrone')
    ax.set_xscale('log')
    ax.set_xlabel(r'Initial Mass / $M_\odot$')
    ax.set_ylabel(r'Cumulative Mass / $M_\odot$')
    ax.legend()
    
    fig,ax = plt.subplots()
    ax.axhline(0,c='k', linestyle='--',alpha=0.5)
    ax.plot(Mmids, MintKroupa(Mmids,k=BFk,minM=0.08)-np.cumsum(Mmids*int_IMFdiffs), label='minM=0.08')
    ax.plot(Mmids, MintKroupa(Mmids,k=BFk,minM=0.07)-np.cumsum(Mmids*int_IMFdiffs), label='minM=0.07')
    ax.plot(Mmids, MintKroupa(Mmids,k=BFk,minM=0.031)-np.cumsum(Mmids*int_IMFdiffs), label='minM=0.031')
    ax.plot(Mmids, MintKroupa(Mmids,k=BFk,minM=isochrone['Mini'].min())-np.cumsum(Mmids*int_IMFdiffs), label='minM=min on isochrone')
    ax.set_xscale('log')
    ax.set_xlabel(r'Initial Mass / $M_\odot$')
    ax.set_ylabel(r'(Function - Isochrone) Cumulative Mass / $M_\odot$')
    ax.set_title(f'mh={mh},age={age/1e9},min={isochrone["Mini"].min()}')
    ax.legend()
    # I need to set minM to minimum mass on isochrone (0.09) to match 
    # cumulative sum as htis can only be doe from that value.
    # But none level off - there's a large fraction of mass always beyond
    # end of parsec isochrones - I need to extent to check exact normalisation
    
    M = np.exp(np.linspace(np.log(1), np.log(5000)))
    fig,ax = plt.subplots()
    ax.plot(M, MintKroupa(M,minM=0.08), label='minM=0.08')
    ax.plot(M, MintKroupa(M,minM=0.07), label='minM=0.07')
    ax.plot(M, MintKroupa(M,minM=0.031), label='minM=0.031')
    ax.plot(M, MintKroupa(M,minM=isochrone['Mini'].min()), label='minM=min on isochrone')
    ax.set_xscale('log')
    ax.set_xlabel(r'Initial Mass / $M_\odot$')
    ax.set_ylabel(r'Cumulative Mass / $M_\odot$')
    ax.legend()
    
    # takes very long time to level off
    print(noUpperLimTotalMassKroupa(minM=0.09)-MintKroupa(150,minM=0.09))
    # Also >10% of mass is over 150 Msun, Kroupa's max mass
    print(MintKroupa(np.array([10,100,300]), minM=0.09))
    print(MintKroupa(np.array([10,100,300]), minM=0.08))
    
    # assuming known k and minM=0.09, wha is parsec max mass that gives total as 1?
    def fun(x,args):
        residualSq = (MintKroupa(x, minM=args)
                      - 1)**2
        return np.sum(residualSq)
    minM=0.09
    res = optimize.minimize(fun, 150, args=(minM,))
    print(f'Best fit Mmax = {res.x[0]:.8f} for minM={minM}')
    minM=0.08
    res = optimize.minimize(fun, 150, args=(minM,))
    print(f'Best fit Mmax = {res.x[0]:.8f} for minM={minM}')
    # isochrones all start at M=0.09, a value from literature
    # int_IMF has an offset which I should ignore, and k such that
    # total mass =1 when integrated from 0.09 to 220MSun,
    # or if max=150 from min of 0.048
    
    # What I should do if calculate my own number based weights,
    # Choosing my own min and max. 2013 paper says 0.07 and 150,
    # but isochrones all clearly have 0.09 as minimum so may want that
    # I think 0.08 is good - in most literature and isochrone G and c values don't change 
    # much over mass range 0.09-0.14, so probably not missing anything by 
    # going a little lower.
    # upper limit also flexible, as only number fraction 5e-5 between 150 and 220
    # assuming range of 0.08-220, so go with 150 as in 2013 paper
    
    # Finally finally, I want to normalise to a total number of 1.
    print(f'To normalise, k = {1/intKroupa(150, k=1)}')
    assert np.isclose(intKroupa(150,k=normk), 1.0)
    
    #Now testing my functions
    w = calcWeights(isogrid)
    cumulativeW = np.zeros(len(w)) # cumsum along each isochrone
    for i in range(len(indx)-1):
        i1,i2 = indx[i], indx[i+1]
        cumulativeW[i1:i2] = np.cumsum(w[i1:i2])
    fig,ax = plt.subplots()
    ax.scatter(isogrid['Mini'], (isogrid['int_IMF'])*normk/BFk-intKroupa(0.08,minM=0.031), s=0.2, alpha=0.5, label='Isogrid')
    ax.scatter(isogrid['Mini'], intKroupa(isogrid['Mini']), s=0.2, alpha=0.5, label='Function')
    ax.scatter(isogrid['Mini'], cumulativeW, s=0.2, alpha=0.5, label='calcWeights')
    ax.set_xscale('log')
    ax.set_xlabel(r'Initial Mass / $M_\odot$')
    ax.set_ylabel(r'Cumulative IMF')
    ax.legend()
    # looks good
    
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
    From CMD website: "Differences between 2 values of int_IMF give the 
    absolute number of stars occupying that isochrone section per unit mass of
    stellar population initially born" (I think unit mass is 1 Solar mass)
    
    """
    MH_logAge_vals, indices = extractIsochrones(isogrid)
    diff = np.zeros(len(isogrid))
    diff[1:] = (isogrid['int_IMF'][1:]-isogrid['int_IMF'][:-1])*(normk/BFk)
    # gives all points except lowest mass point in each isochrone, which is always 0.09 apart from MH=0.45 which is 0.1
    # ks change it t my number=1 normalisation 
    # (i.e. total weight of 1 sine morte isochrone is 1)
    diff[indices] = intKroupa(isogrid['Mini'][indices]) # gives remaining points
    return diff


def loadGrid():
    path = os.path.join(dataDir, 'input_data', 'PARSEC_Kroupa_isochrones.dat')
    return ascii.read(path, guess=True)


    
def checkFracAlive(isogrid, bymass=False):
    MH_logAge, indx = extractIsochrones(isogrid)
    # Most massive still-alive star is at second-last place on isochrone
    # as last is a white dwarf
    indx = np.append(indx, len(isogrid))
    for isonum in range(len(indx)-1):
        i1,i2 = indx[isonum], indx[isonum+1]
        isochrone = isogrid[i1:i2]
        maxmass = isochrone['Mini'][-2]
        print(f'\nIsochrone mh = {MH_logAge[isonum,0]}, age = {10**(MH_logAge[isonum,1]-9):.1f}')
        print(f'Number fraction still alive: {intKroupa(maxmass)}')
        print(f'Mass fraction still alive: {MintKroupa(maxmass)/MintKroupa(150)}')
    
if __name__=='__main__':
    testIMF()
    checkFracAlive(loadGrid())
    pass
    
    