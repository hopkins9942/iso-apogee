import numpy as np
import scipy.optimize
import scipy.stats



# Mock dists
bracket = (0,1)
F1 = lambda x: x #uniform
F2 = lambda x: x**2 #linear pdf

def invert(F, yarr):
    xarr = np.zeros_like(yarr)
    for i, y in enumerate(yarr):
        G = lambda x: F(x)-y
        xarr[i] = scipy.optimize.root_scalar(G, bracket=bracket).root
    return xarr

def sample(F, N):
    y = np.random.uniform(0, 1, N)
    return invert(F, y)

def distinguish(F1, F2, N):

    
    Flist = [F1,F2]
    Slist=[sample(F1, N),sample(F2, N)]
    for k in range(4):
        i,j = np.unravel_index(k, (2,2))
        print(scipy.stats.kstest(Slist[i], Flist[j]))
    
    
    

distinguish(F1, F2, 20)