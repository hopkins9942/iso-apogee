import os
import pickle
import numpy as np
import apogee.tools.read as apread
import apogee.select as apsel
from galpy.util import bovy_plot, save_pickles


allstar = apread.allStar(main=True,rmdups=True)
print('total stars in DR16 main sample: '+str(len(allstar)))

savename = os.path.join(os.environ['DATA'], 'APOGEE', 'apocsf-dr16.pkl')
if os.path.exists(savename):
    with open(savename,'rb') as savefile:
        apo= pickle.load(savefile)
else:
    apo= apsel.apogeeCombinedSelect()
    save_pickles(savename,apo)

statindx = apo.determine_statistical(allstar)
print(str(np.sum(statindx))+' stars in the statistical sample for DR16')
