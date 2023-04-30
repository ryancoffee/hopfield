#!/usr/bin/python3

import numpy as np

rng = np.random.default_rng()

def lorentz(x,c,w,a):
    return a/(np.power((x-c),int(2)) + w)

def gauss(x,c,w,a):
    return a*np.exp(-1*np.power((x-x0)/w,int(2)))


def energy(W,s,theta):
    return -0.5 * np.sum(np.inner(W,s)) + np.inner(s,theta)

def closeEnough(x,y,thresh=.95):
    t = thresh*(x.size)
    return np.sum((x==y).astype(int))>t

def pm2bool(x):
    trueinds = np.where(x>0)
    y = np.zeros_like(x,dtype=bool)
    y[trueinds] = True
    return y

def bool2pm(x):
    return (np.copy(x).astype(int)*2)-1

def randomPatterns_bool(npatterns,size,ratio=0.5):
    rando = rng.uniform(0,1<<8,size=(npatterns,size)).astype(np.uint8)
    inds = np.where(rando<(ratio*(1<<8)))
    out = np.zeros((npatterns,size),dtype=bool)
    out[inds] = 1
    return out

def randomPatterns(npatterns,size,ratio=0.5):
    rando = rng.uniform(0,1<<8,size=(npatterns,size)).astype(np.uint8)
    inds = np.where(rando<(ratio*(1<<8)))
    out = np.zeros((npatterns,size),dtype=np.int8)
    out[inds] = 2
    out -= 1
    return out

def selectDistorted_bool(npatterns,orig,ratio=0.1):
    out = np.zeros((npatterns,orig.shape[1]),dtype=bool)
    rando = rng.uniform(0,1<<8,size=out.shape).astype(np.uint8)
    sampleinds = np.random.choice(np.arange(orig.shape[0]),npatterns)
    for i,s in enumerate(sampleinds):
        out[i,:] = np.copy(orig[s,:])
    
    flipinds = np.where(rando<int(ratio*(1<<8)))
    out[flipinds] = ~out[flipinds]
    return out

def selectDistorted(npatterns,orig,ratio=0.1):
    out = np.zeros((npatterns,orig.shape[1]),dtype=np.int8)
    rando = rng.uniform(0,1<<8,size=out.shape).astype(np.uint8)
    sampleinds = np.random.choice(np.arange(orig.shape[0]),npatterns)
    for i,s in enumerate(sampleinds):
        out[i,:] = np.copy(orig[s,:])
    
    flipinds = np.where(rando<int(ratio*(1<<8)))
    out[flipinds] *= -1
    return out
