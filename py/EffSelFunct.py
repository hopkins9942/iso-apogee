print("starting")
import os
print(os.getcwd())
import numpy as np
import apogee.tools.read as apread
import apogee.select as apsel
import matplotlib.pyplot as plt
import pickle
import isochrones as iso
import mwdust
import tqdm

print("imports done")

# Units:


_DEGTORAD = (np.pi/180.)
_RADTODEG = (180./np.pi)

dmap = mwdust.Combined19('2MASS H')
isogrid = iso.newgrid()

print("set dustmap and isogrid")

# load the selection function
if os.path.exists('../sav/apodr16_csf.dat'):
    with open('../sav/apodr16_csf.dat', 'rb') as f:
        apo = pickle.load(f)
else:
    apo = apsel.apogeeCombinedSelect(year=7)
    with open('../sav/apodr16_csf.dat', 'wb') as f:
        pickle.dump(apo, f)

print("selection function loaded")
def pixelate_and_get_extinction(sample_l, sample_b, sample_d, dmap):
    #highest resolution in the map
    maxnside = np.max(dmap._pix_info['nside'])
    #number of pixels in whole sky...
    npix = healpy.pixelfunc.nside2npix(maxnside)
    sample_pixels = healpy.pixelfunc.ang2pix(maxnside, sample_l, sample_b, lonlat=True)
    whichpix = np.unique(sample_pixels)
    distances = []
    sample_inds = []
    nbin = []
    bin_l, bin_b = healpy.pixelfunc.pix2ang(maxnside, whichpix, lonlat=True)
    extinction = np.zeros(len(sample_l))
    for i in tqdm.tqdm(range(len(whichpix))):
        mask = sample_pixels == whichpix[i]
        distances = sample_d[mask]
        ah = dmap(bin_l[i], bin_b[i], distances)
        extinction[mask] = ah
    return extinction



# pick a random field:
locations = apo.list_fields()
location_index = 765
loc = locations[location_index]
glonGlat = apo.glonGlat(loc)
solid_angle = apo.area(loc)*_DEGTORAD**2

# number of samples to use for the effective selection:
Neffsamp = 1000 #3000

# maximum distance
dist=1

# distance, age and metallicity grid:
ds = np.linspace(0, dist, 101)[1:] # drops point at 0 distance
fehbins = np.linspace(-1., 0.5, 7)
#agebins = np.linspace(-1., 0.5, 7) # check units here

effgrid = np.zeros((len(fehbins)-1,len(ds)))
jkmin = apo.JKmin(loc)

print("about to loop fe bins")

#loop and compute this in each bin of Fe/H
for i in tqdm.tqdm(range(len(fehbins)-1)):
    mask = ((isogrid['logg'] > 1) &
	    (isogrid['logg'] < 3) &
	    (isogrid['MH'] > fehbins[i]) &
	    (isogrid['MH'] < fehbins[i+1])
	   )
    # Add in colour cut here to match selection function?
    effsel_samples = iso.sampleiso(Neffsamp, isogrid[mask], newgrid=True)
    apof = apsel.apogeeEffectiveSelect(apo, dmap3d=dmap,
		MH=effsel_samples['Hmag'],
		JK0=effsel_samples['Jmag']-effsel_samples['Ksmag'])
    effgrid[i] = apof(loc, ds)
print("fe bins looped")
print(np.mean(effgrid, axis=1)) # this is not integrated function, just a check
