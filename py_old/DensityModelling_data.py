import DensityModelling_defs as dm

import apogee.tools.read as apread

import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u

import isochrones

#Plot distributions in colour, magnitude, metallicity, alpha and age
# Compare obs distribution to distributions of effVol
# Compare my own distributions - does hole in xz match up to gap in b for mod(l)<90?
# do a test fit with logNuSun to check it works
# Work out how to deal with cuts and bad stars - think about total mass - then fit real data!

allStar, statIndx = dm.get_allStar_statIndx()

print(len(allStar))
print("Stat sample size: ", np.count_nonzero(statIndx))

statSample = allStar[statIndx]

# print(statSample.dtype.names)

mu, D, R, modz, gLon, gLat, x, y, z, FeH, MgFe, age = dm.extract(statSample)

savePath = "/data/phys-galactic-isos/sjoh4701/APOGEE/outputs/DM_data/"

if False:
	fig, ax = plt.subplots()
	ax.hist2d([l if l<180 else l-360 for l in gLon],gLat, bins=[100,50])
	ax.set_xlabel("l/deg")
	ax.set_ylabel("b/deg")
	fig.savefig(savePath+"glonglat.png")

	x_bins = np.linspace(-20,5,100)
	y_bins = np.linspace(-10,10,80)
	z_bins = np.linspace(-5,5,40)
	D_bins = np.linspace(0,20,80)
	logD_bins = np.logspace(np.log10(0.0001), np.log10(20),80)
	mu_bins = np.linspace(-5,25,80)
	R_bins = np.linspace(0,20,80)
	logR_bins = np.logspace(np.log10(0.1), np.log10(20),80)
	FeH_bins = np.linspace(-1,0.5,50)
	MgFe_bins = np.linspace(-0.2,0.4,50)
	age_bins = np.linspace(-1,14,50)

	fig, ax = plt.subplots()
	ax.hist2d(x,y,bins=[x_bins,y_bins],norm=mpl.colors.LogNorm())
	ax.set_aspect("equal")
	ax.set_xlabel("x/kpc")
	ax.set_ylabel("y/kpc")
	ax.set_title("log(number in statistical sample)")
	fig.savefig(savePath+"xy.png")

	fig, ax = plt.subplots()
	ax.hist2d(x,z,bins=[x_bins,z_bins],norm=mpl.colors.LogNorm())
	ax.set_aspect("equal")
	ax.set_xlabel("x/kpc")
	ax.set_ylabel("z/kpc")
	ax.set_title("log(number in statistical sample)")
	fig.savefig(savePath+"xz.png")
	# do as corner plot?

	fig, ax = plt.subplots()
	ax.hist(D, bins=D_bins)
	ax.set_xlabel("D/kpc")
	fig.savefig(savePath+"D.png")

	fig, ax = plt.subplots()
	ax.hist(D, bins=logD_bins)
	ax.set_xlabel("D/kpc")
	ax.set_xscale('log')
	fig.savefig(savePath+"logD.png")

	fig, ax = plt.subplots()
	ax.hist(mu, bins=mu_bins)
	ax.set_xlabel("mu")
	fig.savefig(savePath+"mu.png")

	fig, ax = plt.subplots()
	ax.hist(R, bins=R_bins)
	ax.set_xlabel("R/kpc")
	fig.savefig(savePath+"R.png")

	fig, ax = plt.subplots()
	ax.hist(R, bins=logR_bins)
	ax.set_xlabel("R/kpc")
	ax.set_xscale('log')
	fig.savefig(savePath+"logR.png")

	fig, ax = plt.subplots()
	ax.hist(z, bins=z_bins)
	ax.set_xlabel("z/kpc")
	fig.savefig(savePath+"z.png")


	fig, ax = plt.subplots()
	ax.hist(FeH, bins=FeH_bins)
	ax.set_xlabel("[Fe/H]")
	fig.savefig(savePath+"FeH.png")

	fig, ax = plt.subplots()
	ax.hist(MgFe, bins=MgFe_bins)
	ax.set_xlabel("[Mg/Fe]")
	fig.savefig(savePath+"MgFe.png")

	fig, ax = plt.subplots()
	ax.hist2d(FeH,MgFe,bins=[FeH_bins,MgFe_bins]) # ,norm=mpl.colors.LogNorm())
	ax.set_aspect("equal")
	ax.set_xlabel("[Fe/H]")
	ax.set_ylabel("[Mg/Fe]")
	fig.savefig(savePath+"MgFeFeH.png")

	fig, ax = plt.subplots()
	ax.hist(age, bins=age_bins)
	ax.set_xlabel("age/Gyr")
	fig.savefig(savePath+"age.png")


# Cuts:
# If D=nan, true dist is probably outside wanted range, therefore can ignore completely as won't be counted in EffSelFunct
# If FeH<-9999 but D!=nan, label as bad because proabably would affect EffSelFunct (sim for MgFe)
# If age outside range 0-14, make either 0 or 14, this is due to dr16 rescaling and this fix is fine as long as I use large age bins.
# Make fit with good stars, then, correct galaxywide numbers in each bin by distributing bad stars proportional to raw numbers and nuSun.
# Then distribute stars over bins, then fit to this. I think all of these should be the same or similar.
# Assumes D is either accurate or nan, rate at which bad abundances are measured is independant of abundance or distance, and any age measured to be outside range 0-14 is actually in nearest bin

mu_min = dm.muMin #4.0
mu_max = dm.muMax #17.0

in_mu_range = np.logical_and.reduce([(mu_min<=mu), (mu<mu_max)])
print("Num in mu range: ", np.count_nonzero(in_mu_range))

data_array = np.zeros([len(dm.FeHBinEdges_array), 3])
for i in range(len(dm.FeHBinEdges_array)):
    in_bin = np.logical_and.reduce([(dm.FeHBinEdges_array[i][0]<=FeH),
                              (FeH<dm.FeHBinEdges_array[i][1]),
                              in_mu_range])
    # jth value is True when jth star lies in mu range and ith bin
    data_array[i][0] = np.count_nonzero(in_bin)
    data_array[i][1] = R[in_bin].mean()
    data_array[i][2] = modz[in_bin].mean() #double check this
print("Unadjusted data: ", data_array)

bad_indices = np.logical_and.reduce([(FeH<-9999), in_mu_range])
# jth value is True, when star should be in my sample, but FeH measurement failed

print("Number of bad FeH stars: ", np.count_nonzero(bad_indices))

print(FeH[FeH<-3]) # lots, all -9999.99
print("THIS ONE: ",FeH[FeH>0.625]) # none
print(np.argwhere(np.isnan(FeH))) # none
print(D[D<0]) # none
print(D[D>100]) # a couple over 100 kpc
print(np.argwhere(np.isnan(D))) # several, need to cut on these
print(np.count_nonzero(np.isnan(D)))
print(np.argwhere(np.isnan(mu))) #several
print(np.count_nonzero(np.isnan(mu))) #same as D as no D<0
print(MgFe[MgFe<-3]) 
print(MgFe[MgFe>3]) # none
print(np.argwhere(np.isnan(MgFe))) # none
#print(age[age<0]) # lots
#print(age[age>50]) # none
print(np.argwhere(np.isnan(age))) #none


# put cut on errors on dist and others?

# N_redgiants to mass
isogrid = isochrones.newgrid()

NRG2mass_array = np.zeros(dm.Nbins)
for i in range(dm.Nbins):
    FeHBinEdges = dm.FeHBinEdges_array[i]
    binMask = ((isogrid['MH'] >  FeHBinEdges[0])
              &(isogrid['MH'] <= FeHBinEdges[1]))
    RGbinMask = binMask & (isogrid['logg'] > 1) & (isogrid['logg'] < 3)
    meanMass = np.average(isogrid[binMask]['Mass'], weights=isogrid[binMask]['weights']) # Check with Ted this is right mass, check units
    print(meanMass)
    RGfraction = isogrid[RGbinMask]['weights'].sum()/isogrid[binMask]['weights'].sum()
    print(RGfraction)
    NRG2mass_array[i] = meanMass/RGfraction
print(NRG2mass_array)


# metallicity to composition
#BB20_data = np.loadtxt("/data/phys-galactic-isos/sjoh4701/APOGEE/T100moleculedata")
#print(BB20_data)

