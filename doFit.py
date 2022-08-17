import numpy as np

from . import utils
from . import pickleGetters # testing relative imports. try putting __init__.py in, or just use .pth file and import as normal. Advantage is not worrying about module name conflicts

Ncpus = int(sys.argv[2])

jobIndex = int(sys.argv[1])

apo = pickleGetters.get_apo()
del apo._specdata, apo._photdata, apo.apo1sel._specdata, apo.apo1sel._photdata, apo.apo2Nsel._specdata, apo.apo2Nsel._photdata, apo.apo2Ssel._specdata, apo.apo2Ssel._photdata
# deletes parts of apo not needed to save memory as apo gets deep copied for each process

binDict = dm.binsToUse[JOB_INDEX]
print(binDict)
print(len(isoUtils.binsToUse))

mu = isoUtils.arr(isoUtils.muGridParams)
print(mu)
D = dm.mu2D(mu)
