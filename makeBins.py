import os
import numpy as np

import astropy.coordinates as coord
import astropy.units as u

from myUtils import binsToUse, clusterDataDir, binName, muMin, muMax, GC_frame
import pickleGetters
import isochrones

def main():
    binList = binsToUse

    for binDict in binList:
        # creates bin directory
        path = os.path.join(clusterDataDir, 'bins', binName(binDict))
        if not os.path.exists(path):
            os.mkdirs(path)
        
        # NRG2mass
        NRG2mass = calculateNRG2mass(binDict)
        NRG2massPath = os.path.join(path, 'NRG2mass.dat')
        with open(NRG2massPath, 'wb') as f:
            pickle.dump(NRG2mass, f)
        print(NRG2mass)

    # data
    calculateData(binList)
    
    # weird order/layout due to use of old code, but it's not that bad
    # may make more sense if all set up is done first, then in one loop over bins pickles are made


# isochrone functs
def calculateNRG2mass(binDict):
    isogrid = isochrones.newgrid()
    whole_bin_mask = calc_isogrid_mask(binDict, isogrid, RG_only=False)
    RG_bin_mask    = calc_isogrid_mask(binDict, isogrid, RG_only=True)
    meanMass = np.average(isogrid[whole_bin_mask]['Mass'], weights=isogrid[whole_bin_mask]['weights'])
    RGfraction = isogrid[RG_bin_mask]['weights'].sum()/isogrid[whole_bin_mask]['weights'].sum()
    return meanMass/RGfraction

def calc_isogrid_mask(binDict, isogrid, RG_only=True): # could be put into isochrones.py
    if RG_only:
        mask = ((1<=isogrid['logg']) & (isogrid['logg']<3))
    else:
        mask = np.full(len(isogrid), True)

    for label, limits in binDict.items():
        field, funct = isogridFieldAndFunct(label)
        if field!='unused_in_isochrones':
            mask &= ((limits[0]<=funct(isogrid[field])) & (funct(isogrid[field])<limits[1]))
        else: pass
    return mask

def isogridFieldAndFunct(label):
    """returns field in ischrone grids given my label in binDict. Add to if needed.
    if undefined in isochrone grid returns empty string. Beware of age
    vs logAge
    Thought: could additionally output a function which maps isogrid value to same scale/units as my values
    Would be better as a match/switch, but installing python 3.10 can be a faff
    """
    if label=='FeH':
        field = 'MH'
        funct = lambda x: x
    elif label=='MgFe':
        field = 'unused_in_isochrones'
        funct = lambda x: np.nan
    elif label=='age':
        field = 'logAge'
        funct = lambda x: 10**(x-9)
    else:
        field = 'UNEXPECTED'
        funct = lambda x: np.nan
    return field, funct



# data functs
def calculateData(bins):
    """
    Calculates data for fit
    """
    allStar, statIndx = pickleGetters.get_allStar_statIndx()

    statSample = allStar[statIndx]
    mu, D, R, modz, gLon, gLat, x, y, z, FeH, MgFe, age = extract(statSample)
    in_mu_range = (muMin<=mu) & (mu<muMax)
    # array of bools with True where star is in mu range
    Nstars = np.count_nonzero(in_mu_range)
    if ('MgFe' in bins[0].keys()): # bit sketchy, assumes all binDicts have same keys and FeH will always be one of them
        print('MgFe used')
        bad_indices = ((FeH<-9999)|(MgFe<-9999)) & in_mu_range
    else:
        print('MgFe not used')
        bad_indices = ((FeH<-9999) & in_mu_range)
    # jth value of bad_indices is True, when star should be in my sample based on mu range, but FeH (or MgFe, if being used) measurement failed
    Nbad = np.count_nonzero(bad_indices)
    print('Number of bad stars in mu range: ', Nbad)
    print('Num of stars in mu range: ', Nstars)
    adjustment_factor = 1/(1-(Nbad/Nstars))
    print(adjustment_factor)
    adjusted_data = np.zeros([len(bins), 3])
    for i in range(len(bins)):
        in_bin = calc_allStarSample_mask(bins[i], statSample)
        # jth value is True when jth star lies in mu range and ith bin
        adjusted_data[i,0] = np.count_nonzero(in_bin)*adjustment_factor
        adjusted_data[i,1] = R[in_bin].mean()
        adjusted_data[i,2] = modz[in_bin].mean() #double check this
        with open(os.path.join(clusterDataDir, 'bins', binName(bins[i]), 'data.dat'), 'wb') as f:
            pickle.dump(adjusted_data[i,:], f)
    print("Adjusted data: ", adjusted_data)
    return adjusted_data
    
def extract(S):
    gLon = S['GLON']
    gLat = S['GLAT']
    D = S['weighted_dist']/1000 # kpc
    mu = 10 + 5*np.log10(D)
    gCoords = coord.SkyCoord(l=gLon*u.deg, b=gLat*u.deg, distance=D*u.kpc, frame='galactic')
    gCentricCoords = gCoords.transform_to(GC_frame)
    x = gCentricCoords.x.to(u.kpc).value
    y = gCentricCoords.y.to(u.kpc).value
    z = gCentricCoords.z.to(u.kpc).value
    R = np.sqrt(x**2 + y**2)
    modz = np.abs(z)
    FeH = S['FE_H']
    MgFe = S['MG_FE']
    age = S['age_lowess_correct']
    return mu, D, R, modz, gLon, gLat, x, y, z, FeH, MgFe, age

def calc_allStarSample_mask(binDict, sample, in_mu_range_only=True):
    """
    Any subsample of allStar can be inputted (ie statSample) as long as fields are like allStar
    """
    if in_mu_range_only:
        field, funct = allStarFieldAndFunct('mu')
        mask = ((utils.muMin<=funct(sample[field])) & (funct(sample[field])<utils.muMax))
        # probably unhelpfully complicated
        # equivalent to muMin<=10+5*np.log10(sample['weighted_dist']/1000) etc.
    else:
        mask = np.full(len(sample),True)
    # preallocates mask of same length as sample, either with True only for stars in mu range or for all stars

    for label, limits in binDict.items():
        field, funct = allStarFieldAndFunct(label)
        mask &= ((limits[0]<=funct(sample[field])) & (funct(sample[field])<limits[1]))
#        if field!='UKNOWN':
#            mask &= ((limits[0]<=funct(sample[field])) & (funct(sample[field])<limits[1]))
#        else:
#            pass # not sure why this was here, want to catch unexpected fields
    return mask

def allStarFieldAndFunct(label):
    """returns field in allStar given my label. Add to if needed.
    if undefined returns empty string. Beware of age
    vs logAge type fields and units"""
    if label=='D':
        field = 'weighted_dist'
        funct = lambda x: x/1000
    elif label=='mu':
        field = 'weighted_dist'
        funct = lambda x: 10 + 5*np.log10(x/1000)
    elif label=='FeH':
        field = 'FE_H'
        funct = lambda x: x
    elif label=='MgFe':
        field = 'MG_FE'
        funct = lambda x: x
    elif label=='age':
        field = 'age_lowess_correct'
        funct = lambda x: np.maximum(np.minimum(x,13.8),0)
    else:
        field = 'UKNOWN'
        funct = lambda x: None
    return field, funct


if __name__ == '__main__':
    main()
