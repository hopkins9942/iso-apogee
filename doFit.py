import datetime
import time
import sys
import os
import pickle

import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.special import gammaincinv, erf
import corner

import mySetup
import myIsochrones
import pickleGetters

# 20230512: need to change this to use an ESF based on measured astroNN age distribuion 


def main(binNum, plotStuff, verbose, label=''):
    if verbose:
        print(f"\n\nStarting! {datetime.datetime.now()}")
    
    # binNum = 20# int(sys.argv[1])
    binDict = mySetup.binList[binNum]
    
    # ESFweightingNum =0#int(sys.argv[2])
    
    binPath = os.path.join(mySetup.dataDir, 'bins', mySetup.binName(binDict))
    
    with open(os.path.join(binPath, 'data.dat'), 'rb') as f:
        data = pickle.load(f)
    

    print("bin: ", binNum, mySetup.binName(binDict))
    if verbose:
        print("data: ", data)
    
    if data[0]==0:
        # No stars in bin
        print("No stars")

        with open(os.path.join(binPath, label+'fit_results.dat'), 'wb') as f:
            pickle.dump(np.array([-999, -999, -999, -999, -999]), f)
        with open(os.path.join(binPath, label+'fit_sigmas.dat'), 'wb') as f:
            pickle.dump(np.array([-999, -999, -999, -999, -999]), f)
        return True # i.e. "successful fit"
    
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
    
    
    
    muMin = 7.0 
    muMax = 17.0
    muStep = 0.1
    locations = pickleGetters.get_locations()
    mu = mySetup.arr((muMin,muMax, muStep))
    
    
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
    assert R.shape==(len(locations),len(mu))
    

    ESF = np.zeros((len(logAge), len(locations),len(mu)))
    for t in range(14):
        ESF[t,:,:] = np.where((mySetup.minR<R)&(R<mySetup.maxR)&(modz<mySetup.maxmodz),
                              (get_effSelFunc(binDict['FeH'][0]+0.025, logAge[t])
                               +get_effSelFunc(binDict['FeH'][0]+0.075, logAge[t]))/2, #mean of ESFs for the two metallicities
                              0)
        #sets ESF to zero outside range considered
    if binDict['FeH'][0]<-0.55:
        # where ages unreliable, uses mean ESF for each age
        meanESF = ESF.mean(0)
        assert meanESF.shape == R.shape
        ESF = np.tile(meanESF, (14,1,1))
    assert ESF.shape==(len(tau),len(locations),len(mu))
    multiplier = (solidAngles*(D**3)*(muStep)*mySetup.ageStep*ESF*np.log(10)/5)
    
    
    
    def ageFactor(tau0, omega):
        """because age dist is truncated normal (0-14Gyr), with proper normalisation
        dist is ageFactor*exp(-(omega/2)*(tau-tau0)**2).
        
        therefore agefactor**-1 = integral exp(-(omega/2)*(tau-tau0)**2 from tau=0-14"""
        return 1/(np.sqrt(np.pi/(2*omega))*(erf(np.sqrt(omega/2)*(14-tau0)) - erf(-np.sqrt(omega/2)*tau0)))
        
    def B(aR, az, tau0, omega):
        return multiplier*ageFactor(tau0, omega)*np.exp(-aR*(R-mySetup.R_Sun) -az*modz -omega*((tau-tau0)**2)/2)
    
    def Blite(aR, az, tau0, omega):
        """without age factor, as appears in function and to make B-weighted averages and objective function faster """
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
        return np.log(Blite(aR,az,tau0,omega).sum()) + aR*(data[1]-mySetup.R_Sun) + az*data[2] + omega*data[4]/2 - omega*tau0*data[3] + omega*(tau0**2)/2
    
    # def fun_uniform(x):
    #     aR, az = x
    #     return fun((aR,az,7,0.0001))
    
    def jac(x):
        aR, az, tau0, omega = x
        return (        data[1] - (R    * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum() ,
                        data[2] - (modz * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum() ,
                -omega*(data[3] - (tau * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum()),
                0.5*data[4] - tau0*data[3] - ((0.5*tau**2 - tau0*tau)*Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum()
                )
    
    # def jac_uniform(x):
    #     aR, az = x
    #     tau0, omega = (7,0.0001)
    #     return ( data[1] - (R    * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum() ,
    #              data[2] - (modz * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum() )
    
    res = scipy.optimize.minimize(fun=fun, x0=(1/data[1],
                                               1/data[2],
                                               data[3],
                                               1/(data[4]-data[3]**2+0.1)), #0.1 is to stop inf when only one star -  omega =inf when one star
                                  jac=jac, 
                                  bounds=((-1,None),(-1,None),(None,None),(0.0001,1)))
    aR, az, tau0, omega = res.x
    f_peak = res.fun
    covariance = res.hess_inv.todense()
        
    
        
    #jac needed for hess?
    #bounds to stop omega exploding when only one star, 1 chosen as errors in age are about 1Gyr and spacing of grid used to approximate is 1Gyr
    # also to stop aR exploding when one star near high R end if range
    
    if verbose:
        print(res)
        print(res.x)
    
    isSuccess = res.success
    
    logNuSun = np.log(data[0]/B(aR,az, tau0, omega).sum())
    
    if verbose:
        print("results: ", logNuSun, aR, az, tau0, omega)
    
    #when bounds are used, .todense() needs to be added as hess isn't explicity  calculated
    if verbose:
        print("covarience: ", covariance)
    variance = np.diag(covariance)
    
    
    # check this, but I think divide by N^0.5 because cov=inverse hess
    sigmas = np.array([((1/data[0])**(0.5))]+[((variance[i]/data[0])**(0.5)) for i in range(4)])
    
    if verbose:
        print("What's saved:")
        print([logNuSun, aR, az, tau0, omega])
        print(sigmas)
    with open(os.path.join(binPath, label+'fit_results.dat'), 'wb') as f:
        pickle.dump(np.array([logNuSun, aR, az, tau0, omega]), f)
    with open(os.path.join(binPath, label+'fit_sigmas.dat'), 'wb') as f:
        pickle.dump(sigmas, f)
    
    
    #checks
    if verbose:
        print('zero = ', (1 - (R      * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum()/data[1]))
        print('zero = ', (1 - (modz   * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum()/data[2]))
        print('zero = ', (1 - (tau    * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum()/data[3]))
        print('zero = ', (1 - (tau**2 * Blite(aR,az,tau0,omega)).sum()/Blite(aR,az,tau0,omega).sum()/data[4]))
    
    agePoints = (np.arange(14*3)+0.5)/3
    with open(os.path.join(binPath, 'ageHist.dat'), 'rb') as f:
        ageHist = pickle.load(f)
    if verbose:
        print('tau0 = ', tau0, ', observed mean = ', ((ageHist[0]*agePoints)/ageHist[0].sum()).sum())
    
    # human readable text file
    path = os.path.join(binPath, label+'results.txt')
    with open(path, 'w') as f:
        out = (f"data: {data}\n\nresult: \n{res}\n\nsigmas: \n{sigmas}\n\nhess: \n{covariance}\n\n"+
               f"\nWhat's saved:\n{[logNuSun, aR, az, tau0, omega]}\n\n"+
               f"median - peak logNuSun = {np.log(gammaincinv(data[0], 0.5)/data[0])}")
        if isSuccess:
            f.write(out)
        else:
            f.write("AAAARGH IT FAILED!") # easily noticible 
            
            
    if plotStuff==False:
        return True
    # plotting
    # What to plot? corner of p, 2d p/beta of ar,az marginalised over tau0 and omega and vice versa
    # just plot hess
    
    # age dist vs model
    ageFine = np.linspace(0,14, 100)
    fig, ax = plt.subplots()
    ax.bar(agePoints, ageHist[0]/(agePoints[1]-agePoints[0]), (agePoints[1]-agePoints[0]), align='center')
    ax.plot(ageFine, data[0]*ageFactor(tau0, omega)*np.exp(-omega*((ageFine-tau0)**2)/2), color='C1')
    ax.set_title(str(binNum)+": Age distributions")
    fig.set_tight_layout(True)
    path = os.path.join(binPath, label+'ageDist.png')
    fig.savefig(path, dpi=300)
    
    # if sigmas[2]>25:
    #     print('NOT graphing, would cause overflow at edge of plot')
    #     return isSuccess
    ncells = 5 #along each axis
    widthFactor = 2
    widths = widthFactor*sigmas[1:]
    #/np.array([aR_forplot, az_forplot])
    # factor for nice cover, division by aR,az is for width of log(aR),log(az)
    # print(widths)
    # aRArr = aR + np.linspace(-widths[0], widths[0], ncells)
    # azArr = az + np.linspace(-widths[1], widths[1], ncells)
    # tau0Arr = tau + np.linspace(-widths[2], widths[2], ncells)
    # omegaArr = omega + np.linspace(-widths[3], widths[3], ncells)
    param = [aR, az, tau0, omega]
    paramArr = [param[i] + np.linspace(-widths[i], widths[i], ncells) for i in range(len(param))]

    pgrid = np.zeros([len(paramArr[i]) for i in range(len(param))]) #(len(aR,az,tau0,omega))
    # lpgrid = np.zeros((len(laRArr), len(lazArr)))
    # beta =  np.zeros([len(paramArr[i]) for i in range(len(param))])
    assert pgrid.size==ncells**4
    for flatIndex in range(pgrid.size):
        mI = np.unravel_index(flatIndex, pgrid.shape)
        # print(mI)
        # time.sleep(0.2)
        pgrid[mI] = np.exp(-data[0]*(fun((paramArr[0][mI[0]],
                                          paramArr[1][mI[1]],
                                          paramArr[2][mI[2]],
                                          paramArr[3][mI[3]]))-f_peak))
        
            
        # beta[mI] = B(paramArr[0][mI[0]],
        #              paramArr[1][mI[1]],
        #              paramArr[2][mI[2]],
        #              paramArr[3][mI[3]]).sum()
    # print(np.where(np.isnan(pgrid[:,1:,:,:])))
    # print(pgrid)
    # only normalises if no nans, but leaves nans in place for checking
    if not np.any(np.isnan(pgrid)):
        intp = pgrid.sum()*np.prod([widths[i]/ncells for i in range(4)])
        pgrid = pgrid/intp
    
    
    # peaklogNuSun = np.log(data[0]/beta)
    
    #how to get samples for corner? times by number wanted, get poisson of each cell
    # fig = corner.corner(
    
    fig, ax = plt.subplots()
    image = ax.imshow(covariance)
    ax.set_title(str(binNum)+": covariance in aR, az, tau0 and omega")
    fig.colorbar(image, ax=ax)
    fig.set_tight_layout(True)
    path = os.path.join(binPath, label+'covariance.png')
    fig.savefig(path, dpi=300)
    
    fig, ax = plt.subplots()
    image = ax.imshow(covariance/np.sqrt(variance*variance.reshape(((4,1)))))
    ax.set_title(str(binNum)+": correlation in aR, az, tau0 and omega")
    fig.colorbar(image, ax=ax)
    fig.set_tight_layout(True)
    path = os.path.join(binPath, label+'correlation.png')
    fig.savefig(path, dpi=300)
    
    # print(paramArr)
    
    fig, ax = plt.subplots()
    image = ax.imshow(pgrid.sum(axis=(2,3), where=~np.isnan(pgrid)).T, origin='lower',
              extent = (param[0]-widths[0]*(1+1/(ncells-1)), param[0]+widths[0]*(1+1/(ncells-1)),
                        param[1]-widths[1]*(1+1/(ncells-1)), param[1]+widths[1]*(1+1/(ncells-1))),
              aspect='auto')
    ax.axvline(aR-sigmas[1], color='C0', alpha=0.5)
    ax.axvline(aR+sigmas[1], color='C0', alpha=0.5)
    ax.axhline(az-sigmas[2], color='C0', alpha=0.5)
    ax.axhline(az+sigmas[2], color='C0', alpha=0.5)
    ax.set_title(str(binNum)+": posterior marginalised over logNuSun, tau0 and omega")
    ax.set_xlabel('aR')
    ax.set_ylabel('az')
    fig.colorbar(image, ax=ax)
    fig.set_tight_layout(True)

    path = os.path.join(binPath, label+'aRazposterior.png')
    fig.savefig(path, dpi=300)
    
    fig, ax = plt.subplots()
    image = ax.imshow(pgrid.sum(axis=(0,1), where=~np.isnan(pgrid)).T, origin='lower',
              extent = (param[2]-widths[2]*(1+1/(ncells-1)), param[2]+widths[2]*(1+1/(ncells-1)),
                        param[3]-widths[3]*(1+1/(ncells-1)), param[3]+widths[3]*(1+1/(ncells-1))),
              aspect='auto')
    ax.axvline(tau0-sigmas[3], color='C0', alpha=0.5)
    ax.axvline(tau0+sigmas[3], color='C0', alpha=0.5)
    ax.axhline(omega-sigmas[4], color='C0', alpha=0.5)
    ax.axhline(omega+sigmas[4], color='C0', alpha=0.5)
    ax.set_title(str(binNum)+": posterior marginalised over logNuSun, aR and az")
    ax.set_xlabel('tau0')
    ax.set_ylabel('omega')
    fig.colorbar(image, ax=ax)
    fig.set_tight_layout(True)
    path = os.path.join(binPath, label+'tau0omegaposterior.png')
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

    # print('logNuSun at peak = ', logNuSun)
    # print("median - peak logNuSun = ", np.log(gammaincinv(data[0], 0.5)/data[0]))
    # fig, ax = plt.subplots()
    # image = ax.imshow(peaklogNuSun.T, origin='lower',
    #           extent = (aRArr[0], aRArr[-1], azArr[0], azArr[-1]),
    #           aspect='auto')
    # ax.set_title("value of logNuSun at posterior peak")
    # ax.set_xlabel('aR')
    # ax.set_ylabel('az')
    # fig.colorbar(image, ax=ax)
    # fig.set_tight_layout(True)
    # path = os.path.join(binPath, f'peaklogNuSun.png')
    # fig.savefig(path, dpi=300)
    
    # print(pgrid)
    
    return isSuccess
    
    
    
    


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




if __name__=='__main__':

    
    # main(154, True, True, 'test')
    
    for i in range(0,len(mySetup.binList)):
        success = main(i, False, False, 'lowFeHUniform+Rzlim+plotwithoutnan')
        if not success:
            print('ARGH FAILURE AGAIN') # for visibility
            break
    
    # main(int(sys.argv[1]), plotStuff=False)
    #run here


