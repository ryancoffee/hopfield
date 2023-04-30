#!/usr/bin/python3

import numpy as np
from utils import lorentz,gauss

rng = np.random.default_rng()

class Molecule:
    def __init__(self):
        self.nbins = 1<<8
        self.centers = []
        self.widths = []
        self.amps = []
        self.x = np.arange(self.nbins,dtype=int)
        self.y = np.zeros(self.nbins,dtype=float)
        self.csum = np.zeros(self.nbins,dtype=float)

    def setParams(self,c,w,a):
        self.centers = c
        self.widths = w
        self.amps = a
        return self

    def setPDF(self):
        self.y = np.zeros(self.nbins,dtype=float)
        for i,c in enumerate(self.centers):
            self.y += lorentz(self.x,c,self.widths[i],self.amps[i])
        self.csum = np.cumsum(self.y)
        return self

    def printPDF(self):
        _ = [print(" "*int(v) + '+') for v in self.y]
        return self

    def samplePDF(self,nsamples):
        test = rng.uniform(0,self.csum[-1],(nsamples,))
        values = np.interp(test,self.csum,self.x.astype(float))
        return values

    def sampleBINvec(self,target=5):
        binvec = np.zeros(self.nbins,dtype=bool)
        test = rng.uniform(0,self.csum[-1],(target,))
        values = np.interp(test,self.csum,self.x).astype(int)
        binvec[values] = True
        return binvec
