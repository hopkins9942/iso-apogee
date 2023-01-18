
import numpy as np
import matplotlib.pyplot as plt

import pickleGetters as pg


def tests():
    S = pg.get_statSample()
    
    elements = ['O', 'MG', 'SI', 'S', 'CA']
    for label in elements:
        label+='_FE'
        fig, ax = plt.subplots()
        ax.hist(S[label], range=(-2,2))
    
    
    # Plan: stars should group by number of elements mismeasured, allowing for selection 
    

def calculateAlpha(S):
    """
    Bovy2016: we average the abundances [O/H], [Mg/H], [Si/H], [S/H], and 
    [Ca/H] and subtract [Fe/H] to obtain the average α-enhancement [α/Fe]. 
    If no measurement was obtained for one of the five α elements,
    it is removed from the average. 
    
    BUT what I have is just /Fe so use that.
    """
    
    alphaH = sum([10**S[e+'_H'] for e in elements])
    fig, ax = plt.subplots()
    ax.hist(alphaH)
    

def calculateData():
    pass
    


if __name__=='__main__':
    