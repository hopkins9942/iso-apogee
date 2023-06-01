import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u

import mySetup
import pickleGetters as pg
import calculateData

cm0 = mpl.colormaps['Blues']





def main(binNum, label):
    
    binDict = mySetup.binList[binNum]
    binPath = os.path.join(mySetup.dataDir, 'bins', mySetup.binName(binDict))
    
    
    S = pg.get_statSample()
    
    # cuts
    aFe, goodAlpha = calculateData.calculateAlphaFe(S)
    
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
    # goodPos = (mySetup.minR<R)&(R<mySetup.maxR)&(modz<mySetup.maxmodz)&(S['weighted_dist_error']/S['weighted_dist']<0.5)
    goodPos = (0<R)&(R<20)&(modz<mySetup.maxmodz)&(S['weighted_dist_error']/S['weighted_dist']<0.5)

    goodCombined = goodLogg&goodPos
    
    
    age = np.where(S['age_lowess_correct']>0, 
                   np.where(S['age_lowess_correct']<14, S['age_lowess_correct'], 13.999),
                   0.0001)
    
    inBin = ((binDict['aFe'][0]<aFe)&(aFe<binDict['aFe'][1])
            &(binDict['FeH'][0]<FeH)&(FeH<binDict['FeH'][1]))
    
    
    
    if len(S[inBin&goodCombined])==0:
        # No stars in bin
        print("No stars")
        return 0
    
    with open(os.path.join(binPath, label+'fit_results.dat'), 'rb') as f:
        logNuSun, aR, az, tau0, omega = pickle.load(f)
        
    #plotting
    #spatial
    Rgrid = np.linspace(4,12).reshape((-1,1))
    modzgrid = np.linspace(0,5)
    
    fig,ax = plt.subplots()
    ax.scatter(R[inBin&goodCombined], modz[inBin&goodCombined], s=0.1, color='red')
    image = ax.imshow((np.exp(-aR*Rgrid -az*modzgrid)).T, origin='lower',
              extent = (0, 20, 0, 5),
              aspect='auto',
              cmap =cm0)
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'modz')
    # this works. You can see pencilbeam fields, and how density along these fields changes with R and z


if __name__=='__main__':
    main(100, 'lowFeHUniform+Rzlim+plotwithoutnan')
    
    