import myUtils
import pickleGetters
from makeBins import extract
import matplotlib.pyplot as plt

allStar, statIndx = pickleGetters.get_allStar_statIndx()
statSample = allStar[statIndx]
names = ['mu', 'D', 'R', 'modz', 'gLon', 'gLat', 'x', 'y', 'z', 'FeH', 'MgFe', 'age']
result = extract(statSample)
#in_mu_range = np.logical_and.reduce([(myUtils.muMin<=mu), (mu<myUtils.muMax)])

path = '/data/phys-galactic-isos/sjoh4701/APOGEE/outputs/plotEverything/'
for i in range(len(result)):
    fig, ax = plt.subplots()
    ax.hist(result[i])
    ax.set_title(names[i])
    fig.savefig(path+names[i])


# look in py_old/DensityModelling_data for more


