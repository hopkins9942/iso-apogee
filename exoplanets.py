import numpy as np
import os
import pickle


import matplotlib as mpl
import matplotlib.pyplot as plt


import mySetup
import myIsochrones


plotDir = '/Users/hopkinsm/APOGEE/plots/exoplanets/'  # Set this
os.makedirs(plotDir, exist_ok=True)





# definitions - skip to line 283 for start of script

cmap0 = mpl.colormaps['Blues']
cmap01 = mpl.colormaps['Purples']
cmap1 = mpl.colormaps['Greys']#mpl.colormaps['Blues']
cmap2 = mpl.colormaps['hsv']
# colourPalette = mpl.colormaps['tab10'](np.linspace(0.05, 0.95, 10))
colourPalette = [ 'goldenrod','darkslateblue', 'teal', 'red']

plt.rcParams.update({
    "text.usetex": True})



class Galaxy:
    def __init__(self, FeHEdges, aFeEdges, rhoSun, logNuSun, aR, az, tau0, omega, sig_logNuSun, sig_aR, sig_az, sig_tau0, sig_omega, NRG2SMM, data):
        """
        amp etc are arrays with [FeH index, aFe index]
        
        logNuSun is what comes out of the fit, the cartesian space number density of red giants at R=R0, z=0 in each bin
        rhoSun is now multiplied by NRG2NLS, divided by volume of bin, so equals the denisty in space and composition of number density distribution of living stars
        
        """
        self.FeHEdges = FeHEdges
        self.aFeEdges = aFeEdges
        self.rhoSun = rhoSun
        self.logNuSun = logNuSun # logNuSun from fit, for testing
        self.aR = aR
        self.az = az
        self.tau0 = tau0
        self.omega = omega
        self.sig_logNuSun = sig_logNuSun # unsure if useful
        self.sig_aR = sig_aR
        self.sig_az = sig_az
        self.sig_tau0 = sig_tau0
        self.sig_omega = sig_omega
        self.NRG2SMM = NRG2SMM
        self.data = data
        
        self.shape = self.rhoSun.shape
        self.FeHWidths = FeHEdges[1:] - FeHEdges[:-1]
        self.FeHMidpoints = (FeHEdges[1:] + FeHEdges[:-1])/2
        self.aFeWidths = aFeEdges[1:] - aFeEdges[:-1]
        self.aFeMidpoints = (aFeEdges[1:] + aFeEdges[:-1])/2
        self.vols = self.FeHWidths.reshape(-1,1) * self.aFeWidths
        assert self.vols.shape==self.rhoSun.shape
        
        
    def mask(self, N=20):# err part currently completely flawed by negative aR
        return (
                (self.data[0,:,:]         >=N)
                )
    
    @classmethod
    def loadFromBins(cls, FeHEdges=mySetup.FeHEdges, aFeEdges=mySetup.aFeEdges, fitLabel='lowFeHUniform+Rzlim+plotwithoutnan'):
        """
        
        """
        shape = (len(FeHEdges)-1, len(aFeEdges)-1)
        rhoSun = np.zeros(shape)#amp of rho
        logNuSun = np.zeros(shape)#log  numbr desnity of giants
        aR = np.zeros(shape)
        az = np.zeros(shape)
        tau0 = np.zeros(shape)
        omega = np.zeros(shape)
        sig_logNuSun = np.zeros(shape)
        sig_aR = np.zeros(shape)
        sig_az = np.zeros(shape)
        sig_tau0 = np.zeros(shape)
        sig_omega = np.zeros(shape)
        NRG2NLS = np.zeros(shape)
        data = np.zeros((5, *shape))
        
        FeHWidths = FeHEdges[1:] - FeHEdges[:-1]
        aFeWidths = aFeEdges[1:] - aFeEdges[:-1]
        vols = FeHWidths.reshape(-1,1) * aFeWidths
        
        isogrid = myIsochrones.loadGrid()
        
        for i in range(shape[0]):
            isochrones = isogrid[(FeHEdges[i]<=isogrid['MH'])&(isogrid['MH']<FeHEdges[i+1])]

            
            for j in range(shape[1]):
            
                binDir  = os.path.join(mySetup.dataDir, 'bins', f'FeH_{FeHEdges[i]:.3f}_{FeHEdges[i+1]:.3f}_aFe_{aFeEdges[j]:.3f}_{aFeEdges[j+1]:.3f}')
                
                with open(os.path.join(binDir, 'data.dat'), 'rb') as f0:
                    data[:,i,j] = np.array(pickle.load(f0))
                    
                with open(os.path.join(binDir, fitLabel+'fit_results.dat'), 'rb') as f1:
                    logNuSun[i,j], aR[i,j], az[i,j], tau0[i,j], omega[i,j] = pickle.load(f1)
                        
                    
                if FeHEdges[0]>-0.55:
                    with open(os.path.join(binDir, fitLabel+'fit_sigmas.dat'), 'rb') as f2:
                        sig_logNuSun[i,j], sig_aR[i,j], sig_az[i,j], sig_tau0[i,j], sig_omega[i,j] = pickle.load(f2)
                
                else:
                    with open(os.path.join(binDir, fitLabel+'fit_sigmas.dat'), 'rb') as f2:
                        sig_logNuSun[i,j], sig_aR[i,j], sig_az[i,j], sig_tau0[i,j], sig_omega[i,j] = pickle.load(f2)
                        
                if data[0,i,j] !=0:
                    if FeHEdges[i]<-0.55: #low Fe
                        NRG2NLS[i,j] = myIsochrones.NRG2NLS(isochrones, 7, 0.0001) #uses uniform age 
                        # NRG2SMM[i,j] = myIsochrones.NRG2NLS(isochrones, 12, 1) #uses old age to get upper lim 
                    else:
                        NRG2NLS[i,j] = myIsochrones.NRG2NLS(isochrones, tau0[i,j], omega[i,j]) 
                    
                    rhoSun[i,j] = NRG2NLS[i,j]*np.exp(logNuSun[i,j])/vols[i,j]
                else:
                    rhoSun[i,j] = 0
                    
        return cls(FeHEdges, aFeEdges, rhoSun, logNuSun, aR, az, tau0, omega, sig_logNuSun, sig_aR, sig_az, sig_tau0, sig_omega, NRG2NLS, data)
    
    
    def hist(self, R=mySetup.R_Sun, z=mySetup.z_Sun, normalised=False):
        hist = self.rhoSun*np.exp( - self.aR*(R-mySetup.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.hist(R,z)) #assumes bins cover whole distribution
    

    def integratedHist(self, Rlim1=mySetup.minR, Rlim2=mySetup.maxR, zlim=mySetup.maxmodz, normalised=False):
        """integrates R=0 to R and z=-z to z, with default of whole galaxy"""
        hist = ((4*np.pi*self.rhoSun*np.exp(self.aR*mySetup.R_Sun)/(self.aR**2 * self.az))
                * ((1+self.aR*Rlim1)*np.exp(-self.aR*Rlim1) - (1+self.aR*Rlim2)*np.exp(-self.aR*Rlim2))
                * (1 - np.exp(-self.az*zlim)))
            
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.integratedHist())
        

    def zintegratedHist(self, R=mySetup.R_Sun, zlim=mySetup.maxmodz, normalised=False):
        """integrates z=-z to z at given R"""
        hist = (2*self.rhoSun*np.exp(-self.aR*(R-mySetup.R_Sun))/(self.az))*(1 - np.exp(-self.az*zlim))
        if not normalised:
            return hist
        else:
            return hist/sum(self.vols*self.zintegratedHist())
        

    def FeH(self, hist, mask=None):
        """
        From hist in FeH and aFe, integrates over aFe to get FeH alone
        
        mask avoids bins with uncertain parameter values 
        """
        if mask==None:
            mask = self.mask()
        return ((np.where(mask, hist, 0)*self.vols).sum(axis=1))/self.FeHWidths
        
        
        
# #### ISO recipe ####
# FeH_p = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
# fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
# compPoly = np.polynomial.polynomial.Polynomial.fit(FeH_p, fH2O_p, 3)
# FeHLow = FeH_p[0]
# FeHHigh = FeH_p[-1]
# fH2OLow = compPoly(FeH_p[-1])
# fH2OHigh = compPoly(FeH_p[0])
# def comp(FeH):
#     return np.where(FeHLow<=FeH, np.where(FeH<FeHHigh,
#                                           compPoly(FeH),
#                                           fH2OLow), fH2OHigh)
    
# def compInv(fH2O):
#     """inv may not work with array inputs"""
#     if np.ndim(fH2O)==0:
#         val = fH2O
#         if fH2OLow<=val<=fH2OHigh:
#             allroots = (compPoly-val).roots()
#             myroot = allroots[(FeH_p[0]<=allroots)&(allroots<=FeH_p[-1])]
#             assert len(myroot)==1 # checks not multiple roots
#             assert np.isreal(myroot[0])
#             return np.real(myroot[0])
#         else:
#             return np.nan
#     else:
#         returnArray = np.zeros_like(fH2O)
#         for i, val in enumerate(fH2O):
#             if fH2OLow<=val<fH2OHigh:
#                 allroots = (compPoly-val).roots()
#                 myroot = allroots[(FeH_p[0]<=allroots)&(allroots<FeH_p[-1])]
#                 assert len(myroot)==1 # checks not multiple roots
#                 assert np.isreal(myroot[0])
#                 returnArray[i] = np.real(myroot[0])
#             else:
#                 returnArray[i] =  np.nan
#         return returnArray
        
# def compDeriv(FeH):
#     return np.where((FeHLow<=FeH)&(FeH<FeHHigh), compPoly.deriv()(FeH), 0)

# for x in [-0.4+0.0001, -0.2, 0, 0.2, 0.4-0.0001]:
#     assert np.isclose(compInv(comp(x)), x) #checks inverse works
    
# class Distributions:
#     """Contains the binned FeH distributions, and methods to get
#     corresponding smoothed and fH2O distributions."""
    
#     alpha = 1 #normalising factor
    
#     def __init__(self,FeHHist, FeHEdges=mySetup.FeHEdges, perVolume=True, normalised=False, ISONumZIndex=1):
#         self.FeHEdges = FeHEdges
#         self.FeHWidths = self.FeHEdges[1:] - self.FeHEdges[:-1]
#         self.FeHMidpoints = (self.FeHEdges[1:] + self.FeHEdges[:-1])/2
#         self.perVolume = perVolume
#         self.isNormalised = normalised
#         if not normalised:
#             self.FeHHist = FeHHist
#         else:
#             self.FeHHist = FeHHist/np.sum(self.FeHWidths*FeHHist)
#         self.ISONumZIndex = ISONumZIndex
        
#         y = np.append(0, np.cumsum(self.FeHWidths*self.FeHHist))
#         dist = scipy.interpolate.CubicSpline(self.FeHEdges, y, bc_type='clamped', extrapolate=True).derivative()
#         def FeHDistFunc(FeH):
#             # sets dist to zero outside range
#             return np.where((self.FeHEdges[0]<=FeH)&(FeH<self.FeHEdges[-1]), dist(FeH), 0)
#         self.FeHDist = FeHDistFunc
        
#         def ISOsPerFeH(FeH):
#             # return self.alpha*(10**(FeH*ISONumZIndex))*self.FeHDist(FeH)
#             return self.alpha*(((10**FeH)/(1+2.78*0.0207*(10**FeH)))**ISONumZIndex)*self.FeHDist(FeH)

#         lowerCount = scipy.integrate.quad(ISOsPerFeH, FeHHigh, self.FeHEdges[-1])[0] 
#         middleCount = scipy.integrate.quad(ISOsPerFeH, FeHLow, FeHHigh)[0]
#         upperCount = scipy.integrate.quad(ISOsPerFeH, self.FeHEdges[0], FeHLow, limit=200)[0]
#         normFactor = (lowerCount+middleCount+upperCount) if self.isNormalised else 1
#         #Assumes bins entirely capture all stars
#         # quad has problem with EAGLE and some rs
        
        
#         def fH2ODistFunc(fH2O):
#             return ISOsPerFeH(compInv(fH2O))/(normFactor*np.abs(compDeriv(compInv(fH2O))))
        
#         self.fH2ODist = fH2ODistFunc
#         self.counts = (lowerCount/normFactor, middleCount/normFactor, upperCount/normFactor)


def plotter(dist, title, G):
    dist = np.where(G.mask(), dist, np.nan)
    fig, ax = plt.subplots()
    image=ax.imshow(dist.T, origin='lower', aspect='auto',
                          extent=(G.FeHEdges[0], G.FeHEdges[-1], G.aFeEdges[0], G.aFeEdges[-1]),
                          cmap=cmap0,norm=mpl.colors.LogNorm())
    ax.set_xlabel(r'$\mathrm{[Fe/H]}$')
    ax.set_ylabel(r'$\mathrm{[\alpha/Fe]}$')
    ax.set_facecolor("gainsboro")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)


####### START #######

"""
Galaxy class contains a model of the Milky Way between R=4kpc to 12kpc, and
z=-5kpc to 5kpc and can be loaded with the .loadFromBins method.
Probably best left with default arguments.

What is stored are 2D numpy arrays with the values of the parameters of the fit 
for each bin in [Fe/H] and [alpha/Fe]

This is currently the sine morte stellar mass density distribution - changing this 
to the normal stellar number/mass density distribution should be easy though. 
"""
G = Galaxy.loadFromBins()

"""
The .hist, .integratedHist and .zintegratedHist methods take inputs of R and z 
coordinates and return the [alpha/Fe] and [Fe/H] distribution as a 2D histogram at this R and z,
or integrated between these values of R and z. See definitions above for
default arguments, all distances are in kiloparsecs.
"""

distribution_at_Sun = G.hist(mySetup.R_Sun, mySetup.z_Sun)
plotter(distribution_at_Sun, 'Solar Neighbourhood', G)

distribution_outer_disk = G.hist(R=11.5, z=0)
plotter(distribution_outer_disk, 'Outer disk', G)

distribution_inner_disk = G.hist(R=4.5, z=0)
plotter(distribution_inner_disk, 'Inner disk', G)

distribution_thick_disk = G.hist(R=mySetup.R_Sun, z=4)
plotter(distribution_thick_disk, 'Thick disk', G)

distribution_galaxy_average = G.integratedHist(Rlim1=mySetup.minR, Rlim2=mySetup.maxR, zlim=mySetup.maxmodz)
plotter(distribution_galaxy_average, 'integrated', G)


"""
The .FeH method converts a 2D histogram in in [alpha/Fe] and [Fe/H] to just
a 1D historgram in [Fe/H] - should be inputted with the output from .hist, 
.integratedHist, or .zintegratedHist

This method by default masks out bins with fewer than 20 stars observed - it doesn't
change the distribution, and avoids bins with uncertain fits
"""

FeH_at_Sun = G.FeH(G.hist())

fig,ax = plt.subplots()
ax.bar(G.FeHMidpoints, FeH_at_Sun, width = G.FeHWidths)
ax.set_xlabel(r'$\mathrm{[Fe/H]}$')
ax.set_title('at Sun')


FeH_outer_disk = G.FeH(G.hist(R=11.5,z=0))

fig,ax = plt.subplots()
ax.bar(G.FeHMidpoints, FeH_outer_disk, width = G.FeHWidths)
ax.set_xlabel(r'$\mathrm{[Fe/H]}$')
ax.set_title('Outer disk')


FeH_galaxy_average = G.FeH(G.integratedHist(Rlim1=mySetup.minR, Rlim2=mySetup.maxR, zlim=mySetup.maxmodz))

fig,ax = plt.subplots()
ax.bar(G.FeHMidpoints, FeH_galaxy_average, width = G.FeHWidths)
ax.set_xlabel(r'$\mathrm{[Fe/H]}$')
ax.set_title('integrated')

"""
The Distributions class smooths the binned FeH distribution and contains an ISO recipe,
I've included it (commented out) but it's probably easier to define your own way
of mapping the FeH distribution to the exoplanet population. Good luck!
"""















