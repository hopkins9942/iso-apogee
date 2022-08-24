import DensityModelling_defs as dm
import pickle


edges = dm.arr((-1.025,0.475,0.1))

FeHBinEdges_array = [[edges[i],edges[i+1]] for i in range(len(edges)-1)]

dm.postProcessing(FeHBinEdges_array, '/home/sjoh4701/APOGEE/iso-apogee/sav/Results20220618/')

