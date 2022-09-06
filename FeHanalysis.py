import os
from functools import partial
import pickle

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.integrate import quad
import matplotlib.pyplot as plt
import git

import myUtils

# for 1D bins - assumed ordered, non-overlapping, equal size

SMOOTH_FeH=True
#FINE=True
POLYDEG=3

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha[:7]
print(repo)
print(sha)


saveDir = f'/Users/hopkinsm/Documents/APOGEE/plots/{sha}/{POLYDEG}-{SMOOTH_FeH}/'

binsDir = '/Users/hopkinsm/data/APOGEE/bins'


def main():
    
    _FeH_edges = myUtils.arr((-1.975, 0.725, 0.1)) #0.625-09.725 has no APOGEE statsample stars, -1.575--1.475 has about 130
    binList = [{'FeH': (_FeH_edges[i], _FeH_edges[i+1])} for i in range(len(_FeH_edges)-1)]
    
    FeHedges = np.append([binList[i]['FeH'][0] for i in range(len(binList))],
                         binList[-1]['FeH'][1])
    FeHmidpoints = (FeHedges[:-1] + FeHedges[1:])/2
    FeHwidths = FeHedges[1:] - FeHedges[:-1]
    
    FeHnumberlogA  = np.zeros(len(binList))
    aR       = np.zeros(len(binList))
    az       = np.zeros(len(binList))
    NRG2mass = np.zeros(len(binList))
    
    for i, binDict in enumerate(binList):
        FeHnumberlogA[i], aR[i], az[i] = loadFitResults(binDict)
        NRG2mass[i] = loadNRG2mass(binDict)
        
    #NRGDM = densityModel(FeHEdges, np.exp(FeHnumberlogA)/FeHWidths, aR, az)
    StellarMassDM = densityModel(FeHedges, NRG2mass*np.exp(FeHnumberlogA)/FeHwidths, aR, az)
    
    #assert StellarMassDM.hist()[0] == StellarMassDM.hist()[-1] == 0
    # asserts that bins cover all data
    
    FeHhist_Sun = StellarMassDM.hist()
    # FeHhist_GalCentre = StellarMassDM.hist(position=(0,0))
    # FeHhist_integrated = StellarMassDM.integratedHist()
    
    print('Check hist')
    print(FeHhist_Sun)
    
    # fH2Oedges = myUtils.arr((0.0, 0.6, (0.005 if FINE else 0.05)))
    # fH2Omidpoints = (fH2Oedges[:-1] + fH2Oedges[1:])/2
    # fH2Owidths = fH2Oedges[1:] - fH2Oedges[:-1]
    
    # fH2Odist_Sun = getfH2Ohist(fH2Oedges, FeHedges, FeHdist_Sun)
    # fH2Odist_GalCentre = getfH2Ohist(fH2Oedges, FeHedges, FeHdist_GalCentre)
    # fH2Odist_integrated = getfH2Ohist(fH2Oedges, FeHedges, FeHdist_integrated)
    
    # smoothfH2Odist_Sun = getSmoothfH2Ohist(fH2Oedges, FeHedges, FeHdist_Sun)
    # smoothfH2Odist_GalCentre = getSmoothfH2Ohist(fH2Oedges, FeHedges, FeHdist_GalCentre)
    # smoothfH2Odist_integrated = getSmoothfH2Ohist(fH2Oedges, FeHedges, FeHdist_integrated)
    
    
    # fH2Odist_Sun = SM2ISOdist(hist2callable(FeHedges, FeHhist_Sun))
    # fH2Odist_GalCentre = SM2ISOdist(hist2callable(FeHedges, FeHhist_GalCentre))
    # fH2Odist_integrated = SM2ISOdist(hist2callable(FeHedges, FeHhist_integrated))
    
    # smoothfH2Odist_Sun = SM2ISOdist(hist2dist(FeHedges, FeHhist_Sun))
    # smoothfH2Odist_GalCentre = SM2ISOdist(smoothDist(FeHedges, FeHhist_GalCentre))
    # smoothfH2Odist_integrated = SM2ISOdist(smoothDist(FeHedges, FeHhist_integrated))
    
    
    #smoothMassSun = smoothDist(FeHedges, FeHdist_Sun)
    
    
    # FeHdist_Sun = hist2dist(FeHedges, FeHhist_Sun)
    # FeHdist_GalCentre = hist2dist(FeHedges, FeHhist_GalCentre)
    # FeHdist_integrated = hist2dist(FeHedges, FeHhist_integrated)
    
    # fH2Odist_Sun, fH2Olower_Sun, fH2Oupper_Sun = SM2ISO(FeHdist_Sun)
    # fH2Odist_GalCentre = SM2ISO(FeHdist_GalCentre)[0]
    # fH2Odist_integrated = SM2ISO(FeHdist_integrated)[0]
    
    
    #plotting
    FeH_plot = np.linspace(-1.5, 0.9, 15*20+1)
    fH2O_plot = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
    
    
    fig, ax = plt.subplots()
    ax.plot(FeH_plot, comp(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,0.1], color=f'C{i}')
    # ax.set_xlim([-1,0.5])
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel(r'$f_\mathrm{H_2O}$')
    saveFig(fig,'comp.png')
    
    fig, ax = plt.subplots()
    ax.plot(FeH_plot, compDeriv(FeH_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([FeH,FeH], [0,-0.1], color=f'C{i}')
    # ax.set_xlim([-1,0.5])
    ax.set_xlabel(r'[Fe/H]')
    ax.set_ylabel(r'$\frac{\mathrm{d}f_\mathrm{H_2O}}{\mathrm{d[Fe/H]}}$')
    saveFig(fig,'compDeriv.png')
    
    fig, ax = plt.subplots()
    ax.plot(fH2O_plot, compInv(fH2O_plot))
    for i, FeH in enumerate(FeHedges):
        ax.plot([comp(FeH),comp(FeH)], [0,0.1], color=f'C{i}') #colours correspond on multiple graphs
    ax.set_xlabel(r'fH2O')
    saveFig(fig, 'compInv.png')
    
    
    # #at Sun
    # fig, ax = plt.subplots()
    # ax.bar(FeHmidpoints, FeHhist_Sun, width = FeHwidths, alpha=0.5)
    # ax.plot(FeH_plot, FeHdist_Sun(FeH_plot), color='C1')
    # for i, FeH in enumerate(FeHedges):
    #     ax.plot([FeH,FeH], [0,1e7], color=f'C{i}')
    # ax.set_xlabel(r'[Fe/H]')
    # ax.set_ylabel('Stellar mass distribution (dM/dVdFeH)')
    # ax.set_title('Solar neighborhood')
    # saveFig(fig, 'SunMass.png')
    
    # # fig, ax = plt.subplots()
    # # ax.bar(FeHmidpoints, np.cumsum(FeHwidths*FeHhist_Sun), width = FeHwidths, alpha=0.5)
    # # ax.plot(FeH_plot, hist2cumulativeDist(FeHedges, FeHhist_Sun)(FeH_plot), color='C1')
    
    # fig,ax = plt.subplots()
    # ax.bar(FeHmidpoints,
    #        [quad(FeHdist_Sun, FeHedges[i], FeHedges[i+1])[0]/FeHwidths[i]
    #             for i in range(len(FeHmidpoints))],
    #        width = FeHwidths)
    # for i, FeH in enumerate(FeHedges):
    #     ax.plot([FeH,FeH], [0,1e7], color=f'C{i}')
    # ax.set_xlabel(r'[Fe/H]')
    # ax.set_ylabel('Stellar mass distribution (dM/dVdFeH)')
    # ax.set_title('Solar neighborhood, checking smoothing')
    # saveFig(fig, 'checkSunMass.png')
    # print('Check spline method:')
    # print(np.array([quad(FeHdist_Sun, FeHedges[i], FeHedges[i+1])[0]/FeHwidths[i]
    #                     for i in range(len(FeHmidpoints))]) - FeHhist_Sun)
    
    # fig, ax = plt.subplots()
    # ax.plot(fH2O_plot, fH2Odist_Sun(fH2O_plot))
    # ax.text(0.1, 1e8, f"lower: {fH2Olower_Sun:.2e}")
    # ax.text(0.4, 1e8, f"upper: {fH2Oupper_Sun:.2e}")
    # for i, FeH in enumerate(FeHedges):
    #     ax.plot([comp(FeH),comp(FeH)], [0,1e8], color=f'C{i}') #colours correspond on multiple graphs
    # ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    # ax.set_ylabel('ISO distribution (dN/dVdfH2O)')
    # ax.set_title('Solar neighborhood')
    # saveFig(fig, 'SunISOs.png')
    
    
    # # At Gal Centre
    # fig, ax = plt.subplots()
    # ax.bar(FeHmidpoints, FeHhist_GalCentre, width = FeHwidths, alpha=0.5)
    # ax.plot(FeH_plot, FeHdist_GalCentre(FeH_plot), color='C1')
    # for i, FeH in enumerate(FeHedges):
    #     ax.plot([FeH,FeH], [0,1e9], color=f'C{i}')
    # ax.set_xlabel(r'[Fe/H]')
    # ax.set_ylabel('Stellar mass distribution (dM/dVdFeH)')
    # ax.set_title('Galactic Centre')
    # saveFig(fig, 'GalCentreMass.png')
    
    # fig, ax = plt.subplots()
    # ax.plot(fH2O_plot, fH2Odist_GalCentre(fH2O_plot))
    # for i, FeH in enumerate(FeHedges):
    #     ax.plot([comp(FeH),comp(FeH)], [0,1e9], color=f'C{i}') #colours correspond on multiple graphs
    # ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    # ax.set_ylabel('ISO distribution (dN/dVdfH2O)')
    # ax.set_title('Galactic Centre')
    # saveFig(fig, 'GalCentreISOs.png')
    
        
    # # Integrated
    # fig, ax = plt.subplots()
    # ax.bar(FeHmidpoints, FeHhist_integrated, width = FeHwidths, alpha=0.5)
    # ax.plot(FeH_plot, FeHdist_integrated(FeH_plot), color='C1')
    # for i, FeH in enumerate(FeHedges):
    #     ax.plot([FeH,FeH], [0,1e10], color=f'C{i}')
    # ax.set_xlabel(r'[Fe/H]')
    # ax.set_ylabel('Stellar mass distribution (dM/dFeH)')
    # ax.set_title('Integrated')
    # saveFig(fig, 'integratedMass.png')
    
    # fig, ax = plt.subplots()
    # ax.plot(fH2O_plot, fH2Odist_integrated(fH2O_plot))
    # for i, FeH in enumerate(FeHedges):
    #     ax.plot([comp(FeH),comp(FeH)], [0,1e11], color=f'C{i}') #colours correspond on multiple graphs
    # ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    # ax.set_ylabel('ISO distribution (dN/dfH2O)')
    # ax.set_title('Integrated')
    # saveFig(fig, 'integratedISOs.png')
    
    
    # # EAGLE
    EAGLE_data = np.loadtxt('/Users/hopkinsm/data/APOGEE/input_data/EAGLE_MW_L0025N0376_REFERENCE_ApogeeRun_30kpc_working.dat') 
    # List of star particles
    EAGLE_mass = EAGLE_data[:,9]
    EAGLE_FeH = EAGLE_data[:,14]
    EAGLEedges = myUtils.arr((-2.975, 1.025, 0.1))
    # EAGLEmidpoints = (EAGLEedges[:-1] + EAGLEedges[1:])/2
    EAGLEwidths = EAGLEedges[1:] - EAGLEedges[:-1]
    FeHhist_EAGLE = np.histogram(EAGLE_FeH, bins=EAGLEedges, weights=EAGLE_mass/EAGLEwidths[0])[0]
    
    # print('FeHhist_EAGLE')
    # print(FeHhist_EAGLE)
    # FeHdist_EAGLE = hist2dist(EAGLEedges, FeHhist_EAGLE)
    # fH2Odist_EAGLE = SM2ISO(FeHdist_EAGLE)[0]

    # EAGLEFeH_plot = np.linspace(-3, 1, 40*20+1)
    
    # fig, ax = plt.subplots()
    # ax.bar(EAGLEmidpoints, FeHhist_EAGLE, width=EAGLEwidths, alpha=0.5)
    # ax.plot(EAGLEFeH_plot, FeHdist_EAGLE(EAGLEFeH_plot), color='C1')
    # ax.set_xlabel('[Fe/H]')
    # ax.set_ylabel('stellar mass distribution (dM/d[Fe/H])')
    # ax.set_title('EAGLE')
    # saveFig(fig,'EAGLEMass.png')

    # fig, ax = plt.subplots()
    # ax.plot(fH2O_plot, fH2Odist_EAGLE(fH2O_plot))
    # ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    # ax.set_ylabel('ISO distribution (dN/df H2O)')
    # ax.set_title('EAGLE')
    # saveFig(fig,'eagleISOs.png')
    
    
    APOedges = np.append([binList[i]['FeH'][0] for i in range(len(binList))],
                         binList[-1]['FeH'][1])
    
    namesList = ['Local', 'GalCentre', 'Integrated', 'EAGLE']
    FeHedgesList = [APOedges, APOedges, APOedges, EAGLEedges]
    FeHhistsList = [StellarMassDM.hist(),
                    StellarMassDM.hist((0,0)),
                    StellarMassDM.integratedHist(),
                    FeHhist_EAGLE
                    ]
    perVolume = [True, True, False, False]
    
    for plotNum in range(4): # to save repeating similar code
        name = namesList[plotNum]
        FeHedges = FeHedgesList[plotNum]
        FeHmidpoints = (FeHedges[:-1] + FeHedges[1:])/2
        FeHwidths = FeHedges[1:] - FeHedges[:-1]
        
        FeHhist = FeHhistsList[plotNum]
        FeHplotPoints = np.linspace(FeHedges[0], FeHedges[-1], 10*len(FeHwidths))
        FeHdist = hist2dist(FeHedges, FeHhist)
        
        fH2OplotPoints = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
        fH2Odist, lowerCount, upperCount = SM2ISO(FeHdist)
        middleCount = quad(fH2Odist, fH2OplotPoints[0], fH2OplotPoints[-1])[0]
        #LBMHighCount = quad(fH2Odist, 0.4, fH2OplotPoints[-1])[0] +upperCount
        
        FeHunit = (r'$\mathrm{M}_\odot \mathrm{dex}^{-1} \mathrm{kpc}^{-3}$' if perVolume[plotNum] 
                   else r'$\mathrm{M}_\odot \mathrm{dex}^{-1}$')
        fH2Ounit = (r'$\mathrm{ISOs} \; \mathrm{kpc}^{-3}$' if perVolume[plotNum] 
                    else r'$\mathrm{ISOs}$')
        fH2OintUnit = (r'$\mathrm{ISOs} \; \mathrm{kpc}^{-3}$' if perVolume[plotNum] 
                    else r'$\mathrm{ISOs}$')
        
        fH2Oheight = fH2Odist(0.3)*4
        
        
        fig, ax = plt.subplots()
        ax.bar(FeHmidpoints, FeHhist, width = FeHwidths, alpha=0.5)
        ax.plot(FeHplotPoints, FeHdist(FeHplotPoints), color='C1')
        for i, FeH in enumerate(FeHedges):
            ax.plot([FeH,FeH], [0,FeHdist(0)/10], color=f'C{i}')
        ax.set_xlabel(r'[Fe/H]')
        ax.set_ylabel(f'Stellar mass distribution ({FeHunit})')
        ax.set_title(name)
        saveFig(fig, f'{name}Mass.png')
        
        print(f'Check spline method: {plotNum}')
        print(np.array([quad(FeHdist, FeHedges[i], FeHedges[i+1])[0]/FeHwidths[i]
                            for i in range(len(FeHmidpoints))]) - FeHhist)
        
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        ax[0].plot(fH2OplotPoints, fH2Odist(fH2OplotPoints))
        # ax.text(fH2Olow, fH2Oheight*0.9, f"lower: {lowerCount:.2e} {fH2OintUnit}")
        # ax.text(fH2Olow, fH2Oheight*0.8, f"middle: {middleCount:.2e} {fH2OintUnit}")
        # ax.text(fH2Olow, fH2Oheight*0.7, f"upper: {upperCount:.2e} {fH2OintUnit}")
        for i, FeH in enumerate(FeHedges):
            ax[0].plot([comp(FeH),comp(FeH)], [0,fH2Oheight/10], color=f'C{i}') #colours correspond on multiple graphs
        # ax[0].set_ylim(0,fH2Oheight)
        ax[0].set_xlabel(r'$f_\mathrm{H_2O}$')
        ax[0].set_ylabel(f'ISO distribution ({fH2Ounit})')
        ax[1].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  [lowerCount, middleCount, upperCount])
        ax[1].set_ylabel(f'ISO distribution ({fH2OintUnit})')
        fig.suptitle(f'{name}')
        saveFig(fig, f'{name}ISOs.png')
        
    # comparisons
    comparisonIndices = [(0,2), (2,3), (0,1)] #Integrated vs solar neighborhood, Integrated vs EAGLE
    for count in range(len(comparisonIndices)):
        p1, p2 = comparisonIndices[count]
        FeHedges1 = FeHedgesList[p1]
        FeHedges2 = FeHedgesList[p2]
        
        FeHplotPoints = (np.linspace(APOedges[0], APOedges[-1], 10*(len(APOedges)-1)) if count!=1
                         else np.linspace(EAGLEedges[0], EAGLEedges[-1], 10*(len(EAGLEedges)-1)))
        FeHdist1 = hist2dist(FeHedges1, FeHhistsList[p1], normalised=True)
        FeHdist2 = hist2dist(FeHedges2, FeHhistsList[p2], normalised=True)
        name = namesList[p1]+'+'+namesList[p2]
        
        fig, ax = plt.subplots()
        ax.plot(FeHplotPoints, FeHdist1(FeHplotPoints), label=namesList[p1])
        ax.plot(FeHplotPoints, FeHdist2(FeHplotPoints), label=namesList[p2])
        ax.legend()
        ax.set_xlabel(r'[Fe/H]')
        ax.set_ylabel(r'Normalised SM distribution ($\mathrm{dex}^{-1}$)')
        ax.set_title(name)
        saveFig(fig, f'{name}Mass.png')
        
        fH2OplotPoints = np.linspace(fH2Olow+0.0001, fH2Ohigh-0.0001)
        fH2Odist1, lowerCount1, upperCount1 = SM2ISO(FeHdist1, normalised=True)
        middleCount1 = quad(fH2Odist1, fH2OplotPoints[0], fH2OplotPoints[-1])[0]
        fH2Odist2, lowerCount2, upperCount2 = SM2ISO(FeHdist2, normalised=True)
        middleCount2 = quad(fH2Odist2, fH2OplotPoints[0], fH2OplotPoints[-1])[0]
        
        fH2Oheight = max(fH2Odist1(0.3)*4, fH2Odist2(0.3)*4)
        
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        ax[0].plot(fH2OplotPoints, fH2Odist1(fH2OplotPoints), label=namesList[p1])
        ax[0].plot(fH2OplotPoints, fH2Odist2(fH2OplotPoints), label=namesList[p2])
        # ax[0].set_ylim(0,fH2Oheight)
        ax[0].set_xlabel(r'$f_\mathrm{H_2O}$')
        ax[0].set_ylabel('Normalised ISO distribution')
        ax[0].legend()
        ax[1].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  [lowerCount1, middleCount1, upperCount1], alpha=0.5, label=namesList[p1])
        ax[1].bar([r'$f_\mathrm{H_2O}<0.07$', 'mid', r'$f_\mathrm{H_2O}>0.51$'],
                  [lowerCount2, middleCount2, upperCount2], alpha=0.5, label=namesList[p2])
        ax[1].set_ylabel('Normalised ISO distribution')
        ax[1].legend()
        fig.suptitle(f'{name}')
        saveFig(fig, f'{name}ISOs.png')
        
        
        
        
        
        
        
    
    #Comparisons - need to normalise
    #Integrated vs solar neighborhood
    # fig, ax = plt.subplots()
    # ax.bar(FeHmidpoints, FeHdist_integrated/sum(FeHwidths*FeHdist_integrated), width = FeHwidths, alpha=0.5, label='Integrated')
    # ax.bar(FeHmidpoints, FeHdist_Sun/sum(FeHwidths*FeHdist_Sun), width = FeHwidths, alpha=0.5, label='Local')
    # ax.set_xlabel(r'[Fe/H]')
    # ax.set_ylabel('Stellar mass distribution (normalised)')
    # ax.legend()
    # saveFig(fig,'localIntegratedMass.png')
    
    # fig, ax = plt.subplots()
    # ax.bar(fH2Omidpoints, fH2Odist_integrated/sum(fH2Owidths*fH2Odist_integrated), width=fH2Owidths, alpha=0.5, label='Integrated')
    # ax.bar(fH2Omidpoints, fH2Odist_Sun/sum(fH2Owidths*fH2Odist_Sun),        width=fH2Owidths, alpha=0.5, label='Local')
    # ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    # ax.set_ylabel('ISO distribution (normalised)')
    # ax.legend()
    # saveFig(fig,'localIntegratedISOs.png')
    
    
    # #Integrated vs EAGLE
    # fig, ax = plt.subplots()
    # ax.bar(FeHmidpoints, FeHdist_integrated/sum(FeHwidths*FeHdist_integrated), width = FeHwidths, alpha=0.5, label='Integrated')
    # ax.bar(EAGLEmidpoints, FeHdist_EAGLE/sum(EAGLEwidths*FeHdist_EAGLE), width = EAGLEwidths, alpha=0.5, label='EAGLE')
    # ax.set_xlabel(r'[Fe/H]')
    # ax.set_ylabel('Stellar mass distribution (normalised)')
    # ax.legend()
    # saveFig(fig,'eagleIntegratedMass.png')
    
    # fig, ax = plt.subplots()
    # ax.bar(fH2Omidpoints, fH2Odist_integrated/sum(fH2Owidths*fH2Odist_integrated), width=fH2Owidths, alpha=0.5, label='Integrated')
    # ax.bar(fH2Omidpoints, fH2Odist_EAGLE/sum(fH2Owidths*fH2Odist_EAGLE),        width=fH2Owidths, alpha=0.5, label='EAGLE')
    # ax.set_xlabel(r'$f_\mathrm{H_2O}$')
    # ax.set_ylabel('ISO distribution (normalised)')
    # ax.legend()
    # saveFig(fig,'eagleIntegratedISOs.png')

    


def saveFig(fig, name):
    os.makedirs(saveDir, exist_ok=True)
    path = saveDir+name
    fig.savefig(path, dpi=300)

# Defining comp:
FeH_p = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
fH2O_p = np.array([0.5098, 0.4905, 0.4468, 0.4129, 0.3563, 0.2918, 0.2173, 0.1532, 0.06516])
compPoly = np.polynomial.polynomial.Polynomial.fit(FeH_p, fH2O_p, POLYDEG)
FeHlow = FeH_p[0]
FeHhigh = FeH_p[-1]
fH2Olow = compPoly(FeH_p[-1])
fH2Ohigh = compPoly(FeH_p[0])
if SMOOTH_FeH:
    def comp(FeH):
        return np.where(FeHlow<=FeH, np.where(FeH<FeHhigh,
                                              compPoly(FeH),
                                              fH2Olow), fH2Ohigh)
        
    def compInv(fH2O):
        """inv mayn't work with array inputs"""
        if np.ndim(fH2O)==0:
            val = fH2O
            if fH2Olow<=val<fH2Ohigh:
                allroots = (compPoly-val).roots()
                myroot = allroots[(FeH_p[0]<=allroots)&(allroots<FeH_p[-1])]
                assert len(myroot)==1 # checks not multiple roots
                assert np.isreal(myroot[0])
                return np.real(myroot[0])
            else:
                return np.nan
        else:
            returnArray = np.zeros_like(fH2O)
            for i, val in enumerate(fH2O):
                if fH2Olow<=val<fH2Ohigh:
                    allroots = (compPoly-val).roots()
                    myroot = allroots[(FeH_p[0]<=allroots)&(allroots<FeH_p[-1])]
                    assert len(myroot)==1 # checks not multiple roots
                    assert np.isreal(myroot[0])
                    returnArray[i] = np.real(myroot[0])
                else:
                    returnArray[i] =  np.nan
            return returnArray
            
    def compDeriv(FeH):
        return np.where((FeHlow<=FeH)&(FeH<FeHhigh), compPoly.deriv()(FeH), 0)
            
else:
    """linear interpolation of BB20"""
    comp = lambda FeH: np.interp(FeH, xp=FeH_p, fp=fH2O_p)
    compInv = lambda fH2O: np.interp(fH2O, xp=np.flip(fH2O_p), fp=np.flip(FeH_p), left=np.nan, right=np.nan)
    compDeriv = lambda FeH: (comp(FeH+0.0001)-comp(FeH))/0.0001

for x in [-0.4+0.0001, -0.2, 0, 0.2, 0.4-0.0001]:
    assert np.isclose(compInv(comp(x)), x) #checks inverse works


def loadFitResults(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'fit_results.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

def loadNRG2mass(binDict):
    path = os.path.join(binsDir, myUtils.binName(binDict), 'NRG2mass.dat')
    with open(path, 'rb') as f:
        return pickle.load(f)

# def getfH2Ohist(fH2Oedges, FeHedges, FeHmassHist):
#     alpha = 1 # unknown constant of proportionality between number of ISOs produces and stellar mass
#     fH2Ohist = np.zeros(len(fH2Oedges)-1)
#     fH2Owidth = fH2Oedges[1]-fH2Oedges[0]
#     FeHwidth = FeHedges[1]-FeHedges[0]
#     NpointsPerBin = int(50*(FeHwidth)/(fH2Owidth))
#     # ensures sufficient points in each fH2O bin
#     # (= NpointsPerBin*fH2)width/FeHwidth)
    
#     FeHarray = np.linspace(FeHedges[0], FeHedges[-1], NpointsPerBin*(len(FeHedges)-1), endpoint=False)
#     FeHarray += (FeHarray[1] - FeHarray[0])/2
#     FeHarrayWidth = FeHarray[1]-FeHarray[0]
#     # evenly spaces NpointsPerBin points in each bin offset from edges
#     # note for smooth dist offset not necessary, but probably best kept
#     fH2Oarray = comp(FeHarray)
    
#     for j in range(len(fH2Ohist)):
#         indexArray = np.nonzero((fH2Oedges[j] <= fH2Oarray)&(fH2Oarray < fH2Oedges[j+1]))[0]//NpointsPerBin
#         # finds index of points in FeHarray, integer division changes them to index in FeHmassDistAmp bins
#         fH2Ohist[j] += (alpha*FeHarrayWidth/fH2Owidth) * FeHmassHist[indexArray].sum()
#     return fH2Ohist 

# def getSmoothfH2Ohist(fH2Oedges, FeHedges, FeHmassHist):
#     FeHmassDist = smoothDist(FeHedges, FeHmassHist)
    
#     alpha = 1 # unknown constant of proportionality between number of ISOs produces and stellar mass
#     fH2Ohist = np.zeros(len(fH2Oedges)-1)
#     fH2Owidth = fH2Oedges[1]-fH2Oedges[0]
#     FeHwidth = FeHedges[1]-FeHedges[0]
#     NpointsPerBin = int(50*(FeHwidth)/(fH2Owidth))
#     # ensures sufficient points in each fH2O bin
#     # (= NpointsPerBin*fH2)width/FeHwidth)
    
#     FeHarray = np.linspace(FeHedges[0], FeHedges[-1], NpointsPerBin*(len(FeHedges)-1), endpoint=False)
#     FeHarray += (FeHarray[1] - FeHarray[0])/2
#     FeHarrayWidth = FeHarray[1]-FeHarray[0]
#     # evenly spaces NpointsPerBin points in each bin offset from edges
#     # note for smooth dist offset not necessary, but probably best kept
#     fH2Oarray = comp(FeHarray)
    
#     for j in range(len(fH2Ohist)):
#         indexArray = np.nonzero((fH2Oedges[j] <= fH2Oarray)&(fH2Oarray < fH2Oedges[j+1]))[0]
#         # finds index of points in FeHarray
#         fH2Ohist[j] += (alpha*FeHarrayWidth/fH2Owidth) * FeHmassDist(FeHarray[indexArray]).sum()
#     return fH2Ohist 
    
# def convert(FeHdist): # Add in count on either end, and maybe normalisation
#     alpha = 1
#     func = lambda fH2O: -alpha*FeHdist(compInv(fH2O))/compDeriv(compInv(fH2O))
#     return func

def SM2ISO(FeHdist, alpha=1, normalised=False):
    normFactor = alpha*quad(FeHdist, -np.inf, np.inf, limit=200)[0] if normalised else 1
    ISOdist = lambda fH2O: -alpha*FeHdist(compInv(fH2O))/(normFactor*compDeriv(compInv(fH2O)))
    lowerEndCount = alpha*quad(FeHdist, FeHhigh, np.inf, limit=200)[0]/normFactor
    upperEndCount = alpha*quad(FeHdist, -np.inf, FeHlow, limit=200)[0]/normFactor
    return (ISOdist, lowerEndCount, upperEndCount)
    
# def bindex(edges, value):
#     if value.ndim==0:
#         return np.nonzero(edges <= value)[0][-1]
#     else:
#         return np.array([np.nonzero(edges <= v)[0][-1] for v in value])


# def hist2cumulativeDist(edges, hist):
#     widths = edges[1:] - edges[:-1]
#     y = np.append(0, np.cumsum(widths*hist))
#     dist = CubicSpline(edges, y, bc_type='clamped', extrapolate=True)
#     # dist = PchipInterpolator(edges, y)
#     def distFunc(FeH):
#         return np.where((edges[0]<=FeH)&(FeH<edges[-1]), dist(FeH), 0)
#     return distFunc

def hist2dist(edges, hist, normalised=False):
    widths = edges[1:] - edges[:-1]
    if not normalised:
        y = np.append(0, np.cumsum(widths*hist))
    else:
        y = np.append(0, np.cumsum(widths*hist)/np.sum(widths*hist))
    
    dist = CubicSpline(edges, y, bc_type='clamped', extrapolate=True).derivative()
    # dist = PchipInterpolator(edges, y).derivative() # gives correct boundary conditions (ie flat) if at least one empty bin on either end
    def distFunc(FeH):
        return np.where((edges[0]<=FeH)&(FeH<edges[-1]), dist(FeH), 0)
    return distFunc

    
# def hist2callable(edges, hist):
#     return (lambda value: hist[bindex(edges, value)])
    
class densityModel:
    """distribution of some 'quantity' (number or mass) in volume
    and FeH"""
    def __init__(self, edges, distAmp, aR, az):
        """amp should be"""
        self.edges = edges
        self.widths = edges[1:]-edges[:-1]
        self.midpoints = (edges[:-1]+edges[1:])/2
        self.distAmp = distAmp
        self.aR = aR
        self.az = az
        # print(len(self.edges),
        # len(self.widths),
        # len(self.midpoints),
        # len(self.distAmp),
        # len(self.aR),
        # len(self.az))

    def hist(self, position=(myUtils.R_Sun, myUtils.z_Sun), normalised=False):
        R, z = position
        dist = self.distAmp*np.exp( - self.aR*(R-myUtils.R_Sun) - self.az*np.abs(z))
        if not normalised:
            return dist
        else:
            return dist/sum(self.widths*self.hist(position))
    
    def integratedHist(self, normalised=False):
        dist = 4*np.pi*self.distAmp*np.exp(self.aR*myUtils.R_Sun)/(self.aR**2 * self.az)
        if not normalised:
            return dist
        else:
            return dist/sum(self.widths*self.integratedHist())
        

# def binParamSpaceVol(binDict):
#     size=1
#     for limits in binDict.values():
#         size*=(limits[1]-limits[0])
#     return size
    
    
if __name__=='__main__':
    main()
