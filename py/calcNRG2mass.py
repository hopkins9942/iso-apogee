import DensityModelling_defs as dm
import pickle

bins = dm.binsToUse

for binDict in bins:
    NRG2mass = dm.NRG2mass(binDict)
    path = dm._DATADIR+'bins/'+dm.binName(binDict)+'/NRG2mass.dat'
    with open(path,'wb') as f:
        pickle.dump(NRG2mass,f)
    print(NRG2mass)
