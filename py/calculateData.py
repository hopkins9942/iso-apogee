import DensityModelling_defs as dm
import pickle


edges = dm.arr((-1.025,0.475,0.1))

FeHBinEdges_array = [[edges[i],edges[i+1]] for i in range(len(edges)-1)]

data = dm.calculateData(FeHBinEdges_array)

print(data)
with open(dm._ROOTDIR+'sav/data_20220618.dat', 'wb') as f:
    pickle.dump(data, f)
