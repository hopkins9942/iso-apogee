#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:45:53 2022

@author: hopkinsm
"""

import datetime
import sys
import os
import pickle

import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.special import gammaincinv


import myUtils
import pickleGetters


def main():
    print(f"Starting! {datetime.datetime.now()}")
    
    apo = pickleGetters.get_apo()

    binNum = int(sys.argv[1])

    binList = myUtils.binsToUse

    binDict = binList[binNum]
    
    binPath = os.path.join(myUtils.clusterDataDir, 'bins', myUtils.binName(binDict))
    
    with open(os.path.join(binPath, 'data.dat'), 'rb') as f:
        data = pickle.load(f)
    
    
    if data[0]==0:
        # No stars in bin
        print("No stars")
        with open(os.path.join(binPath, 'noPyro_fit_results.dat'), 'wb') as f:
            pickle.dump([-999, -999, -999], f)
        return 0
    
    # mu, D, R, modz, solidAngles, gLon, gLat, x, y, z = calc_coords(apo)
    # effSelFunc = get_effSelFunc(binDict)
    # multiplier = (solidAngles*(D**3)*(mu[1]-mu[0])*effSelFunc*np.log(10)/5)
    
    # R_modz_multiplier = (R, modz, multiplier)
    
    print("bin: ", myUtils.binName(binDict))
    print("data: ", data)
    
    mu, D, R, modz, solidAngles, gLon, gLat, x, y, z = calc_coords(apo)
    effSelFunc = get_effSelFunc(binDict)
    multiplier = (solidAngles*(D**3)*(mu[1]-mu[0])*effSelFunc*np.log(10)/5)
    
    
    def B(aR, az):
        return multiplier*np.exp(-aR*(R-myUtils.R_Sun) -az*modz)
    
    def fun(x):
        """equal to stuff - ln(p), where p can be either value of total
        posterior in logNuSun, logaR, logaz OR marginal posterior in logaR, logaz
        (they're proportional)
         
         NOTE although input is aR,az, this is for posterior with unifrm priors
         in and distributed in logaR and logaz"""
        aR, az = x
        return np.log(B(aR,az).sum()) + aR*(data[1] - myUtils.R_Sun) + az*data[2]
    
    def jac(x):
        aR, az = x
        return (data[1] - (R * B(aR, az)).sum()/B(aR, az).sum(),
                data[2] - (modz * B(aR, az)).sum()/B(aR, az).sum()
                )
    
    res = scipy.optimize.minimize(fun=fun, x0=(1/data[1], 1/data[2]), jac=jac, bounds=((0,None), (0,None)))
    
    print(res)
    print(res.x)
    
    aR, az = res.x
    
    logNuSun = np.log(data[0]/B(aR,az).sum())
    
    print("results: ", logNuSun, aR, az)
    
    
    
    with open(os.path.join(binPath, 'noPyro_fit_results.dat'), 'wb') as f:
        print("What's saved:")
        print(logNuSun, aR, az)
        pickle.dump([logNuSun, aR, az], f)
    
    # human readable text file
    path = os.path.join(binPath, 'results.txt')
    with open(path, 'w') as f:
        f.write(f"result: \n{res}\n\nWhat's saved:\n{[logNuSun, aR, az]}\n\n\
                median - peak logNuSun = {np.log(gammaincinv(data[0], 0.5)/data[0])}")
    
    
    # plotting
    
    if (aR<0):
        print("NEGATIVE aR WARNING")
        aR = 0.000000000001
        
    if (az<0):
        print("NEGATIVE az WARNING")
        az = 0.000000000001
    
    
    ncells = 30 #along each axis
    widths = [1, 0.5, 0.5] # taken by looking at pyro fits and doubling it ish - tune!
    # lNuArr = logNuSun + np.linspace(-widths[0]/2, widths[0]/2, ncells)
    laRArr = np.log(aR) + np.linspace(-widths[1]/2, widths[1]/2, ncells)
    lazArr = np.log(az) + np.linspace(-widths[2]/2, widths[2]/2, ncells)

    pgrid = np.zeros((len(laRArr), len(lazArr)))
    beta = np.zeros((len(laRArr), len(lazArr)))
    
    for i in range(len(laRArr)):
        for j in range(len(lazArr)):
            pgrid[i,j] = np.exp(-data[0] *fun((laRArr[i], lazArr[j]))) *(widths[1]/ncells)*(widths[2]/ncells)
            beta[i,j] = B(laRArr[i], lazArr[j]).sum()
    pgrid/pgrid.sum()
    # values are marginal posterior over logaR, logaz
    peaklogNuSun = np.log(data[0]/beta)
    
    fig, ax = plt.subplots()
    image = ax.imshow(pgrid.T, origin='lower',
              extent = (laRArr[0], laRArr[-1], lazArr[0], lazArr[-1]))
    ax.set_title("posterior marginalised over logNuSun")
    ax.set_xlabel('ln aR')
    ax.set_ylabel('ln az')
    fig.colorbar(image, ax=ax)
    fig.set_tight_layout(True)
    path = os.path.join(binPath, 'posterior.png')
    fig.savefig(path, dpi=300)
    
    # peak and median value of logNuSun at each aR,az
    print("median - peak logNuSun = ", np.log(gammaincinv(data[0], 0.5)/data[0]))
    fig, ax = plt.subplots()
    image = ax.imshow(peaklogNuSun.T, origin='lower',
              extent = (laRArr[0], laRArr[-1], lazArr[0], lazArr[-1]))
    ax.set_title("value of logNuSun at posterior peak")
    ax.set_xlabel('ln aR')
    ax.set_ylabel('ln az')
    fig.colorbar(image, ax=ax)
    fig.set_tight_layout(True)
    path = os.path.join(binPath, 'peaklogNuSun.png')
    fig.savefig(path, dpi=300)
    
    
    
    
    
def calc_coords(apo):
    """makes mu, D, R, modz, solidAngles, gLon gLat, and galacticentric
    x, y, and z arrays, for ease of use
    units are kpc, and R and modz is for central angle of field.
    rows are for fields, columns are for mu values"""
    locations = apo.list_fields(cohort='all')
    Nfields = len(locations)
    # locations is list of ids of fields with at least completed cohort of
    #  any type, therefore some stars in statistical sample
    mu = myUtils.arr(myUtils.muGridParams)
    D = 10**(-2+0.2*mu)
    gLon = np.zeros((Nfields, 1))
    gLat = np.zeros((Nfields, 1))
    solidAngles = np.zeros((Nfields, 1))
    # This shape allows clever broadcasting in coord.SkyCoord
    for loc_index, loc in enumerate(locations):
        gLon[loc_index,0], gLat[loc_index,0] = apo.glonGlat(loc)
        solidAngles[loc_index,0] = apo.area(loc)*(np.pi/180)**2 # converts deg^2 to steradians
    gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(myUtils.GC_frame)
    x = gCentricCoords.x.value
    y = gCentricCoords.y.value
    z = gCentricCoords.z.value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)
    return mu, D, R, modz, solidAngles, gLon, gLat, x, y, z

def get_effSelFunc(binDict):
    path = os.path.join(myUtils.clusterDataDir, 'bins', myUtils.binName(binDict), 'effSelFunc.dat')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            effSelFunc = pickle.load(f)
    else:
        raise FileNotFoundError("Currently effSelFunc must be calculated seperately")
    return effSelFunc




if __name__=='__main__':
    main()