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

import scipy.optimize


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
        aR, az = x
        return np.log(B(aR,az).sum()) + aR*(data[1] - myUtils.R_Sun) + az*data[2]
    
    def jac(x):
        aR, az = x
        return (data[1] - (R * B(aR, az)).sum()/B(aR, az).sum(),
                data[2] - (modz * B(aR, az)).sum()/B(aR, az).sum()
                )
    
    res = scipy.optimize.minimize(fun=fun, x0=(1/data[1], 1/data[2]), jac=jac)
    
    print(res)
    print(res.x)
    
    aR, az = res.x
    
    logNuSun = np.log(data[0]/B(aR,az).sum())
    
    print("results: ", logNuSun, aR, az)
    
    with open(os.path.join(binPath, 'noPyro_fit_results.dat'), 'wb') as f:
        print("What's saved:")
        print(logNuSun, aR, az)
        pickle.dump([logNuSun, aR, az], f)
    
    

    
    
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
