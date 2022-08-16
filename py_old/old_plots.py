import datetime
print("Starting")
print(datetime.datetime.now())
print('\n')

import os
import numpy as np
import apogee.tools.read as apread
import apogee.select as apsel
import matplotlib.pyplot as plt
import pickle
import isochrones as iso
import mwdust
import tqdm

print("\nImports done")
print(datetime.datetime.now())
print('\n')

# Units:


_DEGTORAD = (np.pi/180.)
_RADTODEG = (180./np.pi)

dmap = mwdust.Combined19('2MASS H')
isogrid = iso.newgrid()

#print(isogrid.keys())
#print(isogrid['logAge'].min(),  np.median(isogrid['logAge']), isogrid['logAge'].max())

print("\nSet dustmap and isogrid")
print(datetime.datetime.now())
print('\n')

# load the selection function
if os.path.exists('../sav/apodr16_csf.dat'):
    print("\nLoading selection function")
    print(datetime.datetime.now())
    print('\n')

    with open('../sav/apodr16_csf.dat', 'rb') as f:
        apo = pickle.load(f)

    print("\nSelection function loaded")
    print(datetime.datetime.now())
    print('\n')

else:
    print("\nCalculating selection function")
    print(datetime.datetime.now())
    print('\n')
    apo = apsel.apogeeCombinedSelect(year=7)
    print("\nSelection function calculated, now saving")
    print(datetime.datetime.now())
    print('\n')
    with open('../sav/apodr16_csf.dat', 'wb') as f:
        pickle.dump(apo, f)
    print("\nSelection function saved")
    print(datetime.datetime.now())
    print('\n')


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
Neffsamp = 5000

# maximum distance
dist=1

# distance, age and metallicity grid:
ds = np.linspace(0, dist, 101)[1:] # drops point at 0 distance
fehbins = np.linspace(-1., 0.5, 6+1)
ageValues = 10**np.unique(isogrid['logAge']) # age is measured in log10(age/yr). TODO add in real numbers.
# Note I don't think this is uniform in logAge: min=8, med=9.7, max=10.08 - but is in linear age
# grid is sparse in age - in binning the way to do it, or should each age be considered separately

print(np.log10(ageValues))

effgrid = np.zeros((len(fehbins)-1, len(ageValues), len(ds)))
jkmin = apo.JKmin(loc)
#mask = (isogrid['logg'] > 1) & (isogrid['logg'] <= 3) & (isogrid['MH'] > -1.) & (isogrid['MH'] <= 0.5) & (isogrid[''] >= jkmin)


print("\nAbout to loop fe+age")
print(datetime.datetime.now())
print('\n')

#loop and compute this in each bin of Fe/H
for i in range(len(fehbins)-1):
    for j in range(len(ageValues)):
        print(f"\nFe bin {i}, age value {j}")
        print(datetime.datetime.now())
        print('\n')
        mask = ((isogrid['logg'] >= 1) &
                (isogrid['logg'] < 3) &
	        (isogrid['MH'] >= fehbins[i]) &
	        (isogrid['MH'] < fehbins[i+1]) &
	        (isogrid['logAge'] >= np.log10(ageValues[j]-10**7)) &
	        (isogrid['logAge'] < np.log10(ageValues[j]+10**7)) &
	        (isogrid['Jmag'] - isogrid['Ksmag'] > jkmin)
	       )
        # Add in colour cut here to match selection function?
        # This may also fix error caused by element of effsel_samples having colour less than selection function minimum colour 
        # Note will need a check that each field uses colour
        if len(isogrid[mask])==0: print('WARNING: NO GRIDPOINTS IN BIN')
        effsel_samples = iso.sampleiso(Neffsamp, isogrid[mask], newgrid=True)
        apof = apsel.apogeeEffectiveSelect(apo, dmap3d=dmap,
        		MH=effsel_samples['Hmag'],
	        	JK0=effsel_samples['Jmag']-effsel_samples['Ksmag'])
        effgrid[i,j,:] = apof(loc, ds)
print("\nFe+age bins looped")
print(datetime.datetime.now())
print('\n')


print(np.mean(effgrid, axis=2)) # this is not integrated function, just a check



#PLOT EVERYTHING
#print("\nAbout to plot everything")
#print(datetime.datetime.now())
#print('\n')

#plot_samples = [iso.sampleiso(200000,isogrid,newgrid=True),
#                iso.sampleiso(200000,isogrid[mask],newgrid=True)]


# plot all
#fig, axArray = plt.subplots(1,2)
#for m in range(2):
#    axArray[m].scatter(plot_samples[m]['Jmag']-plot_samples[m]['Ksmag'], plot_samples[m]['Hmag'], s=0.1)
#    axArray[m].legend()
#    axArray[m].set_ylim(10,-8)
#    axArray[m].set_xlim(-0.4,1.2)
#    axArray[m].set_xlabel(r'$(J-K_S)$')
#    axArray[m].set_ylabel(r'$M_H$')
#fig.savefig("all.pdf")

# plot binned in FE/H
#fig, axArray = plt.subplots(1,2)
#for m in range(2):
#    for i in range(len(fehbins)-1):
#        bin_samples = plot_samples[m][(
#            (plot_samples[m]['MH'] > fehbins[i]) &
#            (plot_samples[m]['MH'] < fehbins[i+1])
#        )]
#        print(len(bin_samples))
#        axArray[m].scatter(bin_samples['Jmag']-bin_samples['Ksmag'], bin_samples['Hmag'], s=0.1, label=f"feh {fehbins[i]}-{fehbins[i+1]}")
#    axArray[m].legend()
#    axArray[m].set_ylim(10,-8)
#    axArray[m].set_xlim(-0.4,1.2)
#    axArray[m].set_xlabel(r'$(J-K_S)$')
#    axArray[m].set_ylabel(r'$M_H$')
#fig.savefig("fehbinned.pdf")

# plot binned in age
#fig, axArray = plt.subplots(1,2)
#for m in range(2):
#    for i in range(len(agebins)-1):
#        bin_samples = plot_samples[m][(
#            (plot_samples[m]['logAge'] > agebins[i]) &
#            (plot_samples[m]['logAge'] < agebins[i+1])
#        )]
#        print(len(bin_samples))
#        axArray[m].scatter(bin_samples['Jmag']-bin_samples['Ksmag'], bin_samples['Hmag'], s=0.1, label=f"logAge {agebins[i]}-{agebins[i+1]}")
#    axArray[m].legend()
#    axArray[m].set_ylim(10,-8)
#    axArray[m].set_xlim(-0.4,1.2)
#    axArray[m].set_xlabel(r'$(J-K_S)$')
#    axArray[m].set_ylabel(r'$M_H$')
#fig.savefig("agebinned.pdf")

# Note: Generally, older and higher Fe/H means redder

#mask = ((isogrid['logg'] >= 1) &
#                (isogrid['logg'] < 3) &
#	        (isogrid['MH'] >= np.min(fehbins)) &
#	        (isogrid['MH'] < np.max(fehbins)) &
#	        #(isogrid['logAge'] >= agebins[j]) &
#	        #(isogrid['logAge'] < agebins[j+1]) &
#	        (isogrid['Jmag'] - isogrid['Ksmag'] > jkmin)
#	       )
#fig, ax = plt.subplots()
#ax.scatter(isogrid[mask]['Jmag'] - isogrid[mask]['Ksmag'], isogrid[mask]['Hmag'], s=0.1) #plot scaatter in age-fe and colour magnitude
#ax.set_xlabel(r'$(J-K_S)$')
#ax.set_ylabel(r'$M_H$')
#fig.savefig('colourmag.pdf')
#
#fig, ax = plt.subplots()
#ax.scatter(isogrid[mask]['MH'], isogrid[mask]['logAge'], s=0.1) #plot scaatter in age-fe and colour magnitude
#ax.set_xlabel('[Fe/H]')
#ax.set_ylabel('log(Age/Gyr)')
#fig.savefig('FeAge.pdf')

# Note: grid is evenly spaced in Fe/H (spacing of 0.05, with a point at 0.00)
# and in age (NOT logAge) with spacing of 10^9 going from 10^8 to 10^8 + 12*10^9
# Beware of defining masks with no points in them
