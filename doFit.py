import datetime
import sys
import os
import pickle

import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.special import gammaincinv, erf


import mySetup
import myIsochrones
import pickleGetters

# 20230512: need to change this to use an ESF based on measured astroNN age distribuion 

def main(binNum, ESFweightingNum):
    print(f"Starting! {datetime.datetime.now()}")
    
    # binNum = 20# int(sys.argv[1])
    binDict = mySetup.binList[binNum]
    
    # ESFweightingNum =0#int(sys.argv[2])
    
    binPath = os.path.join(mySetup.dataDir, 'bins', mySetup.binName(binDict))
    
    with open(os.path.join(binPath, 'data.dat'), 'rb') as f:
        data = pickle.load(f)
    
    print("bin: ", mySetup.binName(binDict))
    print("data: ", data)
    
    if data[0]==0:
        # No stars in bin
        print("No stars")
        with open(os.path.join(binPath, f'w{ESFweightingNum}fit_results.dat'), 'wb') as f:
            pickle.dump(np.array([-999, -999, -999, -999, -999]), f)
        with open(os.path.join(binPath, f'w{ESFweightingNum}fit_sigmas.dat'), 'wb') as f:
            pickle.dump(np.array([-999, -999, -999, -999, -999]), f)
        return 0
    
    logAge = mySetup.logAges
    
    # fix below
    #proabaly better to get MH and age out of binDict and known values
    
    # MH_logAge = myIsochrones.extractIsochrones(myIsochrones.loadGrid())[0]
    # list of MH-age pairs of isochrones. Used with mask to select isochrones
    
    # print(MH_logAge)
    # MHvals = np.unique(MH_logAge[:,0])
    # logAgevals = np.unique(MH_logAge[:,1])
    # print(MHvals)
    # assert len(MHvals)==2
    # print(logAgevals)
    # assert len(logAgevals)==14
    
    # isochroneMask = ((binDict['FeH'][0] <= MH_logAge[:,0])&(MH_logAge[:,0] < binDict['FeH'][1])) #masks for isochrones with MH in range
    # # isochroneIndices = np.arange(len(isochroneMask))[isochroneMask]
    # print(np.arange(len(MH_logAge))[isochroneMask])# prints indices of selectied isochrones
    # assert np.count_nonzero(isochroneMask) == 28 # 14 ages, two metallicities 
    
    # weighting = np.zeros(len(MH_logAge))
    # if ESFweightingNum==0:
    #     #uniform
    #     weighting[isochroneMask] = 1
        
    # elif ESFweightingNum==1:
    #     #young - weight proportional to 13.9-age/Gyr
    #     weighting[isochroneMask] =  13.9-10**(MH_logAge[isochroneMask,1]-9)
        
    # elif ESFweightingNum==2:
    #     #old - weight proportional t0 age
    #     weighting[isochroneMask] =  10**(MH_logAge[isochroneMask,1]-9)
        
    # else:
    #     raise NotImplementedError('define weighting')
    # weighting/=weighting.sum()
    # print(weighting)
    
    
    
    
    
    locations = pickleGetters.get_locations()
    mu = mySetup.arr(mySetup.muGridParams)
    ESF = np.zeros((len(logAge), len(locations),len(mu)))
    for t in range(14):
        ESF[t,:,:] = (get_effSelFunc(binDict['FeH'][0], logAge[t])
                      + get_effSelFunc(binDict['FeH'][1], logAge[t]))/2 #mean of ESFs for the two metallicities
    
    # meanESF = np.zeros((len(locations),len(mu)))
    # for i in np.arange(len(MH_logAge))[isochroneMask]:
    #     ESF = get_effSelFunc(MH_logAge[i,0], MH_logAge[i,1])
        
    #     fig,ax = plt.subplots()
    #     ax.imshow(ESF.T, origin='lower', aspect='auto')
    #     ax.set_title(str(MH_logAge[i,:]))
        
    #     meanESF += ESF*weighting[i]
        
    # fig,ax = plt.subplots()
    # ax.imshow(meanESF.T, origin='lower', aspect='auto')
    # ax.set_title('mean')
    # path = os.path.join(binPath, f'w{ESFweightingNum}meanESF.png')
    # fig.savefig(path, dpi=300)

    tau = (10**(logAge-9)).reshape((-1,1,1))
    D = mySetup.mu2D(mu)
    solidAngles = np.array(pickleGetters.get_solidAngles()).reshape((-1,1))
    multiplier = (solidAngles*(D**3)*(mySetup.muStep)*mySetup.ageStep*ESF*np.log(10)/5)
    
    gLongLat = pickleGetters.get_gLongLat()
    gLon = gLongLat[:,0].reshape((-1,1))
    gLat = gLongLat[:,1].reshape((-1,1)) # allows fancy shit in SkyCoord
    
    gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(mySetup.GC_frame)
    x = gCentricCoords.x.to(u.kpc).value
    y = gCentricCoords.y.to(u.kpc).value
    z = gCentricCoords.z.to(u.kpc).value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)
    assert R.shape==(len(tau), len(locations),len(mu))
    
    
    def ageFactor(tau0, omega):
        """because age dist is truncated normal (0-14Gyr), with proper normalisation
        dist is ageFactor*exp(-(omega/2)*(tau-tau0)**2).
        
        therefore agefactor**-1 = integral exp(-(omega/2)*(tau-tau0)**2 from tau=0-14"""
        return 1/(np.sqrt(np.pi/(2*omega))*(erf(np.sqrt(omega/2)*(14-tau0)) - erf(-np.sqrt(omega/2)*tau0)))
        
    def B(aR, az, tau0, omega):
        return multiplier*ageFactor(tau0, omega)*np.exp(-aR*(R-mySetup.R_Sun) -az*modz -omega*((tau-tau0)**2)/2)
    
    def Blite(aR, az, tau0, omega):
        """without age factor to make B-weighted averages faster """
        return multiplier*np.exp(-aR*(R-mySetup.R_Sun) -az*modz -omega*((tau-tau0)**2)/2)
    
    def fun(x):
        """proportional to (stuff - ln(p))/N, where p can be either value of total
        posterior in logNuSun, logaR, logaz OR marginal posterior in logaR, logaz
        (they're proportional)
         N=effVol is assumed, so effVol=N and logA=logN-log(B.sum) is substituted
         
         NOTE although input is aR,az, this is for posterior with unifrm priors
         in and distributed in logaR and logaz (and other parameters)
         
         updated 17/05/23 to include age"""
        aR, az, tau0, omega = x
        return np.log(B(aR,az,tau0,omega).sum()) + aR*(data[1]-mySetup.R_Sun) + az*data[2] + omega*data[4]/2 - omega*tau0*data[3] + omega*(tau0**2)/2 - np.log(ageFactor(tau0, omega))
    
    def jac(x):
        aR, az, tau0, omega = x
        return (        data[1] - (R    * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum(),
                        data[2] - (modz * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum(),
                -omega*(data[3] - (modz * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum())
                )
    
    res = scipy.optimize.minimize(fun=fun, x0=(1/data[1], 1/data[2], data[3], 1/(data[4]-data[3]**2)), jac=jac) #jac needed for hess?
    
    print(res)
    print(res.x)
    
    aR, az, tau0, omega = res.x
    isSuccess = res.success
    
    logNuSun = np.log(data[0]/B(aR,az, tau0, omega).sum())
    
    print("results: ", logNuSun, aR, az, tau0, omega)
    
    f_peak = res.fun
    hess = np.linalg.inv(res.hess_inv)
    print("hess: ", hess)
    
    sigmas = np.array([((data[0])**(-0.5))]+[((data[0]*hess[i,i])**(-0.5)) for i in range(4)])
    
    
    print("What's saved:")
    print([logNuSun, aR, az, tau0, omega])
    print(sigmas)
    with open(os.path.join(binPath, f'fit_results.dat'), 'wb') as f:
        pickle.dump(np.array([logNuSun, aR, az, tau0, omega]), f)
    with open(os.path.join(binPath, f'fit_sigmas.dat'), 'wb') as f:
        pickle.dump(sigmas, f)
    
    
    
    # plotting
    # aR_forplot = aR
    # az_forplot = az
    # if (aR<=0):
    #     print("NEGATIVE aR WARNING")
    #     # aR_forplot = 0.000000000001
    #     return "NEGATIVE aR"
        
    # if (az<=0):
    #     print("NEGATIVE az WARNING")
    #     # az_forplot = 0.000000000001
    #     return "NEGATIVE az"
    
    
    ncells = 30 #along each axis
    widthFactor = 6
    widths = widthFactor*sigmas[1:] #/np.array([aR_forplot, az_forplot])
    # factor for nice cover, division by aR,az is for width of log(aR),log(az)
    print(widths)
    aRArr = aR_forplot + np.linspace(-widths[0], widths[0], ncells)
    azArr = az_forplot + np.linspace(-widths[1], widths[1], ncells)

    pgrid = np.zeros((len(aRArr), len(azArr)))
    # lpgrid = np.zeros((len(laRArr), len(lazArr)))
    beta = np.zeros((len(aRArr), len(azArr)))
    
    for i in range(len(aRArr)):
        for j in range(len(azArr)):
            pgrid[i,j] = np.exp(-data[0]*(fun((aRArr[i], azArr[j]))-f_peak)) #*(widths[0]/ncells)*(widths[1]/ncells)
            # lpgrid[i,j] = -data[0]*(fun((np.exp(laRArr[i]), np.exp(lazArr[j])))-f_peak)
            beta[i,j] = B(aRArr[i], azArr[j]).sum()
    
    # values are marginal posterior over logaR, logaz (value per log(aR),log(az))
    intp = pgrid.sum()*(widths[0]/ncells)*(widths[1]/ncells)
    pgrid = pgrid/intp
    
    peaklogNuSun = np.log(data[0]/beta)
    
    fig, ax = plt.subplots()
    image = ax.imshow(pgrid.T, origin='lower',
              extent = (aRArr[0], aRArr[-1], azArr[0], azArr[-1]),
              aspect='auto')
    ax.axvline(aR_forplot-sigmas[1], color='C0', alpha=0.5)
    ax.axvline(aR_forplot+sigmas[1], color='C0', alpha=0.5)
    ax.axhline(az_forplot-sigmas[2], color='C0', alpha=0.5)
    ax.axhline(az_forplot+sigmas[2], color='C0', alpha=0.5)
    ax.set_title("posterior marginalised over logNuSun")
    ax.set_xlabel('aR')
    ax.set_ylabel('az')
    fig.colorbar(image, ax=ax)
    fig.set_tight_layout(True)
    path = os.path.join(binPath, f'w{ESFweightingNum}posterior.png')
    fig.savefig(path, dpi=300)
    
    # fig, ax = plt.subplots()
    # image = ax.imshow(lpgrid.T, origin='lower',
    #           extent = (laRArr[0], laRArr[-1], lazArr[0], lazArr[-1]),
    #           aspect='auto')
    # ax.set_title("log(posterior marginalised over logNuSun)")
    # ax.set_xlabel('ln aR')
    # ax.set_ylabel('ln az')
    # fig.colorbar(image, ax=ax)
    # fig.set_tight_layout(True)
    # path = os.path.join(binPath, 'logposterior.png')
    # fig.savefig(path, dpi=300)
    
    # peak and median value of logNuSun at each aR,az
    print('logNuSun at peak = ', logNuSun)
    print("median - peak logNuSun = ", np.log(gammaincinv(data[0], 0.5)/data[0]))
    fig, ax = plt.subplots()
    image = ax.imshow(peaklogNuSun.T, origin='lower',
              extent = (aRArr[0], aRArr[-1], azArr[0], azArr[-1]),
              aspect='auto')
    ax.set_title("value of logNuSun at posterior peak")
    ax.set_xlabel('aR')
    ax.set_ylabel('az')
    fig.colorbar(image, ax=ax)
    fig.set_tight_layout(True)
    path = os.path.join(binPath, f'w{ESFweightingNum}peaklogNuSun.png')
    fig.savefig(path, dpi=300)
    
    # print(pgrid)
    
    # human readable text file
    path = os.path.join(binPath, f'w{ESFweightingNum}results.txt')
    with open(path, 'w') as f:
        out = f"data: {data}\n\nresult: \n{res}\n\nsigmas: \n{sigmas}\n\nhess: \n{hess}\n\nwidths: \n{widths}\n\nWhat's saved:\n{[logNuSun, aR, az]}\n\nmedian - peak logNuSun = {np.log(gammaincinv(data[0], 0.5)/data[0])}"
        if isSuccess:
            f.write(out)
        else:
            f.write("AAAARGH IT FAILED!") # easily noticible 
    
    
    
    
    
# def calc_coords(apo):
#     """makes mu, D, R, modz, solidAngles, gLon gLat, and galacticentric
#     x, y, and z arrays, for ease of use
#     units are kpc, and R and modz is for central angle of field.
#     rows are for fields, columns are for mu values"""
#     locations = apo.list_fields(cohort='all')
#     Nfields = len(locations)
#     # locations is list of ids of fields with at least completed cohort of
#     #  any type, therefore some stars in statistical sample
#     mu = myUtils.arr(myUtils.muGridParams)
#     D = 10**(-2+0.2*mu)
#     gLon = np.zeros((Nfields, 1))
#     gLat = np.zeros((Nfields, 1))
#     solidAngles = np.zeros((Nfields, 1))
#     # This shape allows clever broadcasting in coord.SkyCoord
#     for loc_index, loc in enumerate(locations):
#         gLon[loc_index,0], gLat[loc_index,0] = apo.glonGlat(loc)
#         solidAngles[loc_index,0] = apo.area(loc)*(np.pi/180)**2 # converts deg^2 to steradians
#     gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
#     gCentricCoords = gCoords.transform_to(myUtils.GC_frame)
#     x = gCentricCoords.x.value
#     y = gCentricCoords.y.value
#     z = gCentricCoords.z.value
#     R = np.sqrt(x**2 + y**2)
#     modz = np.abs(z)
#     return mu, D, R, modz, solidAngles, gLon, gLat, x, y, z



def get_effSelFunc(MH, logAge):
    """loads ESF for one isochrone (ie one value of MH, one value of age)"""
    path = os.path.join(mySetup.dataDir, 'ESF', f'MH_{MH:.3f}_logAge_{logAge:.3f}.dat')
    print(path)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            effSelFunc = pickle.load(f)
    else:
        raise FileNotFoundError("Currently effSelFunc must be calculated seperately")
    return effSelFunc




if __name__=='__main__':
    #main(142,0)
    main(int(sys.argv[1]), int(sys.argv[2]))



