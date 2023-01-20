import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u

import mySetup
import pickleGetters as pg


elements = ['O', 'MG', 'SI', 'S', 'CA']

def tests():
    S = pg.get_statSample()
    
    # Showing seperated peaks
    aFe = sum([S[e+'_FE'] for e in elements])/5
    categories = np.digitize(aFe, [-3000, -1000])
    
    fig, ax = plt.subplots()
    ax.hist(aFe[categories==0], bins=100)
    fig, ax = plt.subplots()
    ax.hist(aFe[categories==1], bins=100)
    fig, ax = plt.subplots()
    ax.hist(aFe[categories==2], bins=100)
    
    aFe,goodAlpha = calculateAlphaFe(S)
    fig, ax = plt.subplots()
    ax.hist(aFe[goodAlpha], bins=100)
    
    for label in elements:
        label+='_FE'
        fig, ax = plt.subplots()
        ax.hist(S[S[label]>-1][label],bins=100)
        
    FeH = S['FE_H']
    goodFe = (FeH>-9999)
    fig, ax = plt.subplots()
    ax.hist(FeH[goodFe], bins=100)
    
    fig, ax = plt.subplots()
    ax.hist2d(FeH[goodFe&goodAlpha], aFe[goodFe&goodAlpha], bins=50)
    
    mu = mySetup.D2mu(S['weighted_dist']/1000) #10+5*np.log10(S['weighted_dist']/1000)
    goodMu = (mySetup.muMin<=mu) & (mu<mySetup.muMax)
    goodLogg = (1<=S['LOGG']) & (S['LOGG']<3)
    
    # Logg cut removes dwarves:
    fig, ax = plt.subplots()
    ax.hist(mu, bins=100)
    fig, ax = plt.subplots()
    ax.hist((S['J0']-S['K0']), bins=100)
    
    fig, ax = plt.subplots()
    ax.hist(mu[goodLogg], bins=100)
    fig, ax = plt.subplots()
    ax.hist((S['J0']-S['K0'])[goodLogg], bins=100)
    
    fig, ax = plt.subplots()
    ax.hist(S['H'], bins=100)
    fig, ax = plt.subplots()
    ax.hist(S['H'][goodLogg], bins=100)
    # all H good though some JK0<0.3 - probably to do with dereddening
    
    

def calculateAlphaFe(S):
    """
    Bovy2016: we average the abundances [O/H], [Mg/H], [Si/H], [S/H], and 
    [Ca/H] and subtract [Fe/H] to obtain the average α-enhancement [α/Fe]. 
    If no measurement was obtained for one of the five α elements,
    it is removed from the average. 
    
    BUT what I have is just /Fe so use that.
    
    Bad abundances given value of -9999.99, so simple average puts stars with
    all five measured well in range (-0.2, 0.5), and all with one measured badly 
    in range (-0.2, 0.4)-2000 because equal to 0.8*(meanOf4)-9999.99/5
    
    """
    
    alphaFe = sum([S[e+'_FE'] for e in elements])/5
    categories = np.digitize(alphaFe, [-3000, -1000])
    alphaFe[categories==1] = (5*alphaFe[categories==1] + 9999.99)/4
    goodAlpha = (categories!=0)
    return alphaFe, goodAlpha


def calculateData():
    """
    Cuts on mu and logg 
    """
    S = pg.get_statSample()
    
    # cuts
    aFe, goodAlpha = calculateAlphaFe(S)
    
    FeH = S['FE_H']
    goodFe = (FeH>-9999)
    
    goodLogg = (1<=S['LOGG']) & (S['LOGG']<3)
    D = S['weighted_dist']/1000 # kpc
    mu = mySetup.D2mu(D)
    goodMu = (mySetup.muMin<=mu) & (mu<mySetup.muMax)
    
    goodCombined = goodAlpha & goodFe & goodLogg & goodMu
    
    
    #coords
    gCoords = coord.SkyCoord(l=S['GLON']*u.deg, b=S['GLAT']*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(mySetup.GC_frame)
    x = gCentricCoords.x.to(u.kpc).value
    y = gCentricCoords.y.to(u.kpc).value
    z = gCentricCoords.z.to(u.kpc).value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)
    
    Nstars = np.count_nonzero(goodLogg & goodMu)
    Nbad = np.count_nonzero(np.logical_not(goodAlpha | goodFe) & goodLogg & goodMu)
    print('Num of stars in just mu range: ', np.count_nonzero(goodMu))
    print('Num of stars in mu and logg range: ', Nstars)
    print('Number of stars in mu and logg range missing abundance in either aFe or FeH: ', Nbad)
    print('Number of stars in mu and logg range missing abundance in FeH: ',
          np.count_nonzero(np.logical_not(goodFe) & goodLogg & goodMu))
    
    print('Number of stars in mu and logg range missing abundance in aFe: ',
          np.count_nonzero(np.logical_not(goodAlpha) & goodLogg & goodMu))
    
    print('Number of stars in mu and logg range missing abundance in aFe but has FeH: ',
          np.count_nonzero(np.logical_not(goodAlpha) & goodFe & goodLogg & goodMu))
    
    print('Number of stars in mu and logg range missing abundance in FeH but has aFe: ',
          np.count_nonzero(np.logical_not(goodFe) & goodAlpha & goodLogg & goodMu))
    # all adds up, if FeH missing then so is aFe
    # only 39 stars have FeH but miss aFe, small enough to consider generally bad and distribute as if FeH not known
    
    adjustment_factor = 1/(1-(Nbad/Nstars))
    #assuming stars in mu and logg range but with bad abundaces would be in one of the bins, need to add to make ESF correct
    print(adjustment_factor)
    
    for binDict in mySetup.binList:
        bindices = (
            (binDict['FeH'][0]<=FeH)&(FeH<binDict['FeH'][1])&
            (binDict['aFe'][0]<=aFe)&(aFe<binDict['aFe'][1])&
            (goodLogg & goodMu)
            ) # indices of good stars in bin
        N = np.count_nonzero(bindices)*adjustment_factor
        meanR = R[bindices].mean() if N!=0 else 0
        meanmodz = modz[bindices].mean() if N!=0 else 0
        print(mySetup.binName(binDict), N, meanR, meanmodz)
        with open(os.path.join(mySetup.dataDir, 'bins', mySetup.binName(binDict), 'data.dat'), 'wb') as f:
            pickle.dump(np.array([N, meanR, meanmodz]), f)
    
    
    

if __name__=='__main__':
    calculateData()
    pass



