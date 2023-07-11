import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u


import matplotlib as mpl
cmap1 = mpl.colormaps['Blues']

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
    nPerBin=ax.hist2d(FeH[goodFe&goodAlpha], aFe[goodFe&goodAlpha], bins=50)[0]
    
    mu = mySetup.D2mu(S['weighted_dist']/1000) #10+5*np.log10(S['weighted_dist']/1000)
    goodMu = (mySetup.muMin<=mu) & (mu<mySetup.muMax)
    goodLogg = (1<=S['LOGG']) & (S['LOGG']<3)
    
    # Logg cut removes dwarves:
    
    fig, ax = plt.subplots()
    ax.hist(FeH[goodFe], bins=100)
    ax.hist(FeH[goodFe & goodLogg], bins=100)
    # removing dwarfs reduces sharpness of peak in BB20 range
    
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
    
    fig, ax = plt.subplots()
    totalMHperBin, FeHEdges, aFeEdges, *_ =ax.hist2d(FeH[goodFe&goodAlpha],
                    aFe[goodFe&goodAlpha], bins=50,
              weights=S[goodFe&goodAlpha]['M_H'],
              cmap=cmap1)
    
    fig, ax = plt.subplots(2,1)
    image=ax[1].imshow((totalMHperBin/nPerBin).T, origin='lower', aspect='auto',
                    extent=(FeHEdges[0], FeHEdges[-1], aFeEdges[0], aFeEdges[-1]), 
              cmap=cmap1)
    fig.colorbar(image, cax=ax[0], orientation='horizontal')
    # M_H correlated pretty much exactly with FE_H, with no varience with alpha/FE
    
    
    fig, ax = plt.subplots()
    totalALPHAperBin, FeHEdges, aFeEdges, *_ =ax.hist2d(FeH[goodFe&goodAlpha],
                    aFe[goodFe&goodAlpha], bins=50,
              weights=S[goodFe&goodAlpha]['ALPHA_M'],
              cmap=cmap1)
    
    fig, ax = plt.subplots()
    image=ax.imshow((totalALPHAperBin/nPerBin).T, origin='lower', aspect='auto',
                    extent=(FeHEdges[0], FeHEdges[-1], aFeEdges[0], aFeEdges[-1]), 
              cmap=cmap1)
    fig.colorbar(image)
    # Not quite as strict as M_H vs Fe_H
    # makes sence given ASPCAP procedure - find params fitting whole spectrum with 
    # solar scaled [M/H] and [alpha/M], then for each element vary these only 
    # considereing spectral features sensitive to element in question. Fe has 
    # lots of features so fit well just be [M/H], alpha elements have fewer
    

def plotR():
    S = pg.get_statSample()
    
    goodLogg = (1<=S['LOGG']) & (S['LOGG']<3)
    
    D = S['weighted_dist']/1000 # kpc
    mu = mySetup.D2mu(D)
    goodMu = (mySetup.muMin<=mu) & (mu<mySetup.muMax)
    
    #maps age to 0-14 range
    
    good = goodLogg & goodMu
    
    gCoords = coord.SkyCoord(l=S['GLON']*u.deg, b=S['GLAT']*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(mySetup.GC_frame)
    x = gCentricCoords.x.to(u.kpc).value
    y = gCentricCoords.y.to(u.kpc).value
    z = gCentricCoords.z.to(u.kpc).value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)
    
    fig,ax = plt.subplots()
    ax.hist(R[good&(S['FE_H']>0)], bins=50)
    
    fig,ax = plt.subplots()
    ax.hist(R[good&(S['FE_H']<0)], bins=50)
    
    print(len(R[good&(R>12)])/len(R[good]))
    print(len(R[good&(R<4)])/len(R[good]))
    print(len(R[good&(R<12)&(R>4)])/len(R[good]))

def plotmodz():
    S = pg.get_statSample()
    
    goodLogg = (1<=S['LOGG']) & (S['LOGG']<3)
    
    D = S['weighted_dist']/1000 # kpc
    mu = mySetup.D2mu(D)
    goodMu = (mySetup.muMin<=mu) & (mu<mySetup.muMax)
    
    #maps age to 0-14 range
    
    good = goodLogg & goodMu
    
    gCoords = coord.SkyCoord(l=S['GLON']*u.deg, b=S['GLAT']*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(mySetup.GC_frame)
    x = gCentricCoords.x.to(u.kpc).value
    y = gCentricCoords.y.to(u.kpc).value
    z = gCentricCoords.z.to(u.kpc).value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)
    
    fig,ax = plt.subplots()
    ax.hist(modz[good&(S['FE_H']>0)], bins=50)
    ax.set_xlim((0,5))
    
    fig,ax = plt.subplots()
    ax.hist(modz[good&(S['FE_H']<0)], bins=50)
    ax.set_xlim((0,5))
    
    print(len(modz[good&(modz>1)])/len(modz[good]))
    print(len(modz[good&(modz<1)])/len(modz[good]))
    print(max(modz[good]))

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



def calculateData():#probably bad practoce to have name name as file
    """
    Cuts on mu and logg, distributes stars without abundances over bins to make up numbers
    """
    S = pg.get_statSample()
    
    # cuts
    aFe, goodAlpha = calculateAlphaFe(S)
    
    FeH = S['FE_H']
    goodFe = (FeH>-9999)
    
    goodLogg = (1<=S['LOGG']) & (S['LOGG']<3)

    
    D = S['weighted_dist']/1000 # kpc
    
    #coords
    gCoords = coord.SkyCoord(l=S['GLON']*u.deg, b=S['GLAT']*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(mySetup.GC_frame)
    x = gCentricCoords.x.to(u.kpc).value
    y = gCentricCoords.y.to(u.kpc).value
    z = gCentricCoords.z.to(u.kpc).value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)
    

    
    # mu = mySetup.D2mu(D)
    # goodMu = (mySetup.muMin<=mu) & (mu<mySetup.muMax)
    goodPos = (mySetup.minR<R)&(R<mySetup.maxR)&(modz<mySetup.maxmodz)&(S['weighted_dist_error']/S['weighted_dist']<0.5)
    
    age = np.where(S['age_lowess_correct']>0, 
                   np.where(S['age_lowess_correct']<14, S['age_lowess_correct'], 13.999),
                   0.0001)
    #maps age to 0-14 range
    
    # goodCombined = goodAlpha & goodFe & goodLogg & goodMu
    
    
    
    Nstars = np.count_nonzero(goodLogg & goodPos)
    Nbad = np.count_nonzero(np.logical_not(goodAlpha | goodFe) & goodLogg & goodPos)
    print('Num of stars in just pos range: ', np.count_nonzero(goodPos))
    print('Num of stars in pos and logg range: ', Nstars)
    print('Number of stars in pos and logg range missing abundance in either aFe or FeH: ', Nbad)
    print('Number of stars in pos and logg range missing abundance in FeH: ',
          np.count_nonzero(np.logical_not(goodFe) & goodLogg & goodPos))
    
    print('Number of stars in pos and logg range missing abundance in aFe: ',
          np.count_nonzero(np.logical_not(goodAlpha) & goodLogg & goodPos))
    
    print('Number of stars in pos and logg range missing abundance in aFe but has FeH: ',
          np.count_nonzero(np.logical_not(goodAlpha) & goodFe & goodLogg & goodPos))
    
    print('Number of stars in pos and logg range missing abundance in FeH but has aFe: ',
          np.count_nonzero(np.logical_not(goodFe) & goodAlpha & goodLogg & goodPos))
    # all adds up, if FeH missing then so is goodPos
    # only 39 stars have FeH but miss aFe, small enough to consider generally bad and distribute as if FeH not known
    
    adjustment_factor = 1/(1-(Nbad/Nstars))
    #assuming stars in mu and logg range but with bad abundaces would be in one of the bins, need to add to make ESF correct
    print(adjustment_factor)
    
    for binDict in mySetup.binList:
        bindices = (
            (binDict['FeH'][0]<=FeH)&(FeH<binDict['FeH'][1])&
            (binDict['aFe'][0]<=aFe)&(aFe<binDict['aFe'][1])&

            (goodLogg & goodPos)
            ) # indices of good stars in bin
        N = np.count_nonzero(bindices)*adjustment_factor
        meanR = R[bindices].mean() if N!=0 else 0
        meanmodz = modz[bindices].mean() if N!=0 else 0

        meanage = age[bindices].mean() if N!=0 else 0
        meansquareage = (age[bindices]*age[bindices]).mean() if N!=0 else 0
        # print(mySetup.binName(binDict), N, meanR, meanmodz, meanage, meansquareage)
        with open(os.path.join(mySetup.dataDir, 'bins', mySetup.binName(binDict), 'data.dat'), 'wb') as f:
            pickle.dump(np.array([N, meanR, meanmodz, meanage, meansquareage]), f)
            
        ageHist = np.histogram(age[bindices], bins = 14*3, range=(0,14))
        # print(ageHist)
        with open(os.path.join(mySetup.dataDir, 'bins', mySetup.binName(binDict), 'ageHist.dat'), 'wb') as f:
            pickle.dump(ageHist, f)
        with open(os.path.join(mySetup.dataDir, 'bins', mySetup.binName(binDict), 'ageHist.txt'), 'w') as f:
            f.write(str(ageHist))
    
    
    

if __name__=='__main__':
    calculateData()
    pass



