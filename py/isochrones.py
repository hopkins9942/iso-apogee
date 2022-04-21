import os
os.environ['RESULTS_VERS'] = "l33"
import numpy as np
import apogee.select as apsel
import pickle
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import integrate
import isodist
from isodist import imf as mimf
import tqdm
from astropy.io import ascii

_ROOTDIR = "/home/sjoh4701/APOGEE/iso-apogee/"

def generate_isogrid():
    """
    generate a recarray with all the entries from PARSEC isochrones in isodist
    """
    zs = np.arange(0.0005,0.0605, 0.0005)
    zlist = []
    for i in range(len(zs)):
        zlist.append(format(zs[i],'.4f'))
    iso = isodist.PadovaIsochrone(type='2mass-spitzer-wise', Z=zs, parsec=True)

    logages = []
    mets = []
    js = []
    hs = []
    ks = []
    loggs = []
    teffs = []
    imf = []
    Mimf = []
    deltam = []
    deltaN = []
    M_ini = []
    M_act = []
    logL = []
    iso_logages = iso._logages
    iso_Zs = iso._ZS
    for i in tqdm.tqdm(range(len(iso_logages))):
        for j in range(len(iso_Zs)):
            thisage = iso_logages[i]
            thisZ = iso_Zs[j]
            thisiso = iso(thisage, Z=thisZ)
            ms = np.linspace(0.1, np.max(thisiso['M_ini']), 1000)
            inter = interp1d(ms, np.cumsum(mimf.lognormalChabrier2001(ms)*(ms[1]-ms[0])))
            int_IMF = inter(thisiso['M_ini'])
            inter = interp1d(ms, np.cumsum(ms*mimf.lognormalChabrier2001(ms)*(ms[1]-ms[0])))
            int_mIMF = inter(thisiso['M_ini'])
            so = np.argsort(thisiso['M_ini'])
            loggs.extend(thisiso['logg'][so])
            logages.extend(thisiso['logage'][so])
            mets.extend(np.ones(len(thisiso['H'][so]))*thisZ)
            js.extend(thisiso['J'][so])
            hs.extend(thisiso['H'][so])
            ks.extend(thisiso['Ks'][so])
            teffs.extend(thisiso['logTe'][so])
            imf.extend(int_IMF[so])
            Mimf.extend(int_mIMF[so])
            deltaN.extend(np.concatenate([[int_IMF[so][0],],int_IMF[so][1:]-int_IMF[so][:-1]]))
            deltam.extend(np.concatenate([[int_mIMF[so][0],],int_mIMF[so][1:]-int_mIMF[so][:-1]]))
            M_ini.extend(thisiso['M_ini'][so])
            M_act.extend(thisiso['M_act'][so])
            logL.extend(thisiso['logL'][so])
    logages = np.array(logages)
    mets = np.array(mets)
    js = np.array(js)
    hs = np.array(hs)
    ks = np.array(ks)
    loggs = np.array(loggs)
    teffs = 10**np.array(teffs)
    imf = np.array(imf)
    Mimf = np.array(Mimf)
    deltam = np.array(deltam)
    deltaN = np.array(deltaN)
    M_ini = np.array(M_ini)
    M_act = np.array(M_act)
    logL = np.array(logL)
    rec = np.recarray(len(deltam), dtype=[('logageyr', float),
                                          ('Z', float),
                                          ('J', float),
                                          ('H', float),
                                          ('K', float),
                                          ('logg', float),
                                          ('teff', float),
                                          ('int_IMF', float),
                                          ('int_mIMF', float),
                                          ('deltaM', float),
                                          ('deltaN', float),
                                          ('M_ini', float),
                                          ('M_act', float),
                                          ('logL', float)])

    rec['logageyr'] = logages
    rec['Z'] = mets
    rec['J'] = js
    rec['H'] = hs
    rec['K'] = ks
    rec['logg'] = loggs
    rec['teff'] = teffs
    rec['int_IMF'] = imf
    rec['int_mIMF'] = Mimf
    rec['deltaM'] = deltam
    rec['deltaN'] =deltaN
    rec['M_ini'] = M_ini
    rec['M_act'] = M_act
    rec['logL'] = logL
    return rec

def newgrid():
    table = ascii.read(_ROOTDIR+'sav/PARSEC_lognormChab2001_linearagefeh.dat', guess=True)
    return table


def sampleiso(N, iso, weights='number', newgrid=False):
    """
    Sample isochrone recarray iso weighted by lognormal chabrier (2001) IMF (default in PARSEC)
    """
    if newgrid:
        weight_sort = np.argsort(iso['weights'])
        inds = np.array(range(len(iso)))
        inter = interp1d(np.cumsum(iso['weights'][weight_sort])/np.sum(iso['weights']), inds, kind='nearest')
        random_indices = inter(np.random.rand(N)).astype(int)
        return iso[weight_sort][random_indices]
    else:
        logagekey = 'logageyr'
        zkey = 'Z'
        jkey, hkey, kkey = 'J', 'H', 'K'
        if weights.lower() == 'number':
            weights = iso['deltaN']*(10**(iso[logagekey]-9)/iso[zkey])
        elif weights.lower() == 'mass':
            weights = iso['deltaM']*(10**(iso[logagekey]-9)/iso[zkey])
        sort = np.argsort(weights)
        cdf = integrate.cumtrapz(weights[sort], initial=0)
        tinter = interp1d(cdf/cdf[-1], range(len(weights[sort])), kind='linear')
        randinds = np.round(tinter(np.random.rand(N))).astype(np.int64)
        return iso[sort][randinds]
    
    
def average_mass(iso, weights='number'):
    logagekey = 'logageyr'
    zkey = 'Z'
    if weights.lower() == 'number':
        weights = iso['deltaN']*(10**(iso[logagekey]-9)/iso[zkey])
    elif weights.lower() == 'mass':
        weights = iso['deltaM']*(10**(iso[logagekey]-9)/iso[zkey])
    return np.sum(iso['M_act']*weights)/np.sum(weights)

def fraction_in_range(iso, fulliso, weights='number'):
    logagekey = 'logageyr'
    zkey = 'Z'
    if weights.lower() == 'number':
        weights = iso['deltaN']*(10**(iso[logagekey]-9)/iso[zkey])
        fullweights = fulliso['deltaN']*(10**(fulliso[logagekey]-9)/fulliso[zkey])
    elif weights.lower() == 'mass':
        weights = iso['deltaM']*(10**(iso[logagekey]-9)/iso[zkey])
        fullweights = fulliso['deltaM']*(10**(fulliso[logagekey]-9)/fulliso[zkey])
    return np.sum(weights)/np.sum(fullweights)
