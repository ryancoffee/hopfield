#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import sys

from Hopfield import HopfieldNetwork,HopfieldNetwork_bool,HopfieldNetwork_theta
import utils

def main_bool(sz,ntrain,ntest): # sz = num neurons, ntrain is num of training samples

    patterns = utils.randomPatterns_bool(ntrain,sz,ratio=.25)
    #print('orig:',patterns)
    orig = np.copy(patterns)
    fig,ax = plt.subplots(3,1)
    ax[0].pcolor(orig)
    network = HopfieldNetwork_bool(sz)
    network.train(patterns)
    #print('final',patterns)
    newpatterns = utils.selectDistorted_bool(ntest,orig,ratio=.05)
    ax[1].pcolor(newpatterns) 
    newout = np.zeros(newpatterns.shape,dtype=bool)
    for r,newp in enumerate(newpatterns):
        newout[r,:] = network.recall(newp)
    ax[2].pcolor(newout)
    plt.show()
    return

def main_pm(sz,ntrain,ntest): # sz = num neurons, ntrain is num of training samples

    patterns = utils.randomPatterns(ntrain,sz,ratio=.25)
    orig = np.copy(patterns)
    network = HopfieldNetwork(sz)
    network.train(patterns)
    newpatterns = utils.selectDistorted(ntest,orig,ratio=.25)
    newout = np.zeros(newpatterns.shape,dtype=int)
    for r,newp in enumerate(newpatterns):
        newout[r,:] = network.recall(newp)
    return

def main_theta(sz,ntrain,ntest): # sz = num neurons, ntrain is num of training samples
    patterns = utils.randomPatterns(ntrain,sz,ratio=.25)
    orig = np.copy(patterns)
    fig,ax = plt.subplots(3,1)
    ax[0].pcolor(orig)
    network = HopfieldNetwork_theta(sz)
    network.set_max_iter(25).train(patterns)
    newpatterns = utils.selectDistorted_bool(ntest,orig,ratio=.05)
    ax[1].pcolor(newpatterns) 
    newout = np.zeros(newpatterns.shape,dtype=bool)
    for r,newp in enumerate(newpatterns):
        newout[r,:] = network.recall(newp)
    ax[2].pcolor(newout)
    plt.show()

    return


if __name__ == '__main__':
    print('There is something fishy going on here.  For high neurons, it seems there are only recall of all true or all false, this is a bi-polar Hopfield ;-) ')
    if len(sys.argv)<4:
        print('syntax: runHopfieldy.py <sz|n_neurons> <ntrain> <ntest> <bool|pm|theta>')
    elif len(sys.argv)==5:
        if sys.argv[-1]=='bool':
            main_bool(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
        elif sys.argv[-1]=='pm':
            main_pm(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
        elif sys.argv[-1]=='theta':
            main_theta(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
        else:
            print('Gotta give me a version < bool | pm | theta >')

