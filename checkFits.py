import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
import git

import mySetup
import pickleGetters as pg
import calculateData
import myIsochrones

cmap0 = mpl.colormaps['Blues']
cmap01 = mpl.colormaps['Purples']
cmap1 = mpl.colormaps['Greys']#mpl.colormaps['Blues']
cmap2 = mpl.colormaps['hsv']
# colourPalette = mpl.colormaps['tab10'](np.linspace(0.05, 0.95, 10))
colourPalette = [ 'goldenrod','darkslateblue', 'teal', 'red']

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:7]
plotDir = f'/Users/hopkinsm/APOGEE/plots/{sha}/'
os.makedirs(plotDir, exist_ok=True)


def get_effSelFunc(MH, logAge):
    """loads ESF for one isochrone (ie one value of MH, one value of age)"""
    path = os.path.join(mySetup.dataDir, 'ESF', f'MH_{MH:.3f}_logAge_{logAge:.3f}.dat')
    # print(path)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            effSelFunc = pickle.load(f)
    else:
        raise FileNotFoundError("Currently effSelFunc must be calculated seperately")
    return effSelFunc


    
    
    
    
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
    goodPos = (mySetup.minR<R)&(R<mySetup.maxR)&(modz<mySetup.maxmodz)&(S['weighted_dist_error']/S['weighted_dist']<0.5)

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
    modzgrid = np.linspace(0,2)
    
    fig,ax = plt.subplots()
    ax.scatter(R[inBin&goodCombined], modz[inBin&goodCombined], s=0.1, alpha=0.5, color='red')
    image = ax.imshow((Rgrid*np.exp(-aR*Rgrid -az*modzgrid)).T, origin='lower',#Rgird* becasue this makes it desnity per R per z
              extent = (4, 12, 0, 2),
              aspect='auto',
              cmap=cmap1)
    ax.set_xlabel(r'$R/\mathrm{kpc}$')
    ax.set_ylabel(r'$\vert z\vert /\mathrm{kpc}$')
    ax.set_title('Bin ' +str(binNum))
    # this works. You can see pencilbeam fields, and how density along these fields changes with R and z
    
    fig.set_tight_layout(True)
    path = plotDir+'/'+str(binNum)+'Rmodzdist.pdf'
    fig.savefig(path, dpi=300)
    
    # ESF
    # If i come back to this, move to seprate function, or do for only one bin
    # muMin = 7.0 
    # muMax = 17.0
    # muStep = 0.1
    # locations = pg.get_locations()
    # mu = mySetup.arr((muMin,muMax, muStep))
    # D = mySetup.mu2D(mu)
    # solidAngles = np.array(pg.get_solidAngles()).reshape((-1,1))
    
    # gLongLat = pg.get_gLongLat()
    # gLon = gLongLat[:,0].reshape((-1,1))
    # gLat = gLongLat[:,1].reshape((-1,1)) # allows fancy shit in SkyCoord
    
    # gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
    # gCentricCoords = gCoords.transform_to(mySetup.GC_frame)
    # x = gCentricCoords.x.to(u.kpc).value
    # y = gCentricCoords.y.to(u.kpc).value
    # z = gCentricCoords.z.to(u.kpc).value
    # R = np.sqrt(x**2 + y**2)
    # modz = np.abs(z)
    # assert R.shape==(len(locations),len(mu))
    
    # effSelFunc = np.zeros((len(solidAngles),len(mu)))
    # def weight(MH, logAge):
    #     return np.exp(-0.5*((MH--0.1)/0.5)**2)*np.exp(-0.5*((np.exp(logAge)-7)/2)**2)
    # MH_logAge, _ = myIsochrones.extractIsochrones(myIsochrones.loadGrid())
    # for MH, logAge in MH_logAge:
    #     path = os.path.join(mySetup.dataDir, 'ESF', f'MH_{MH:.3f}_logAge_{logAge:.3f}.dat')
    #     with open(path, 'rb') as f:
    #         effSelFunc += np.where((mySetup.minR<R)&(R<mySetup.maxR)&(modz<mySetup.maxmodz),
    #                               pickle.load(f),#*weight(MH,logAge),
    #                               0)
    
    # fig,ax = plt.subplots()
    # for i in range(len(locations)):
    #     ax.scatter(R[i,:], modz[i,:], color=cmap0(effSelFunc[i,:]), s=0.1)
    # ax.set_xlim(4,12)
    # ax.set_ylim(0,5)
                
    
    
                              
if __name__=='__main__':
    for i in range(25,26):#25*7,29*7):
        main(i, 'lowFeHUniform+Rzlim+plotwithoutnan')
    
    
    
    
    
    
    