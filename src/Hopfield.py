import numpy as np
import matplotlib.pyplot as plt
import sys

from utils import closeEnough,bool2pm,pm2bool

class HopfieldNetwork_bool:
    def __init__(self,sz):
        self.sz = sz
        self.weights = np.zeros((sz,sz),dtype=int)

    def train(self,patterns):
        for pat in bool2pm(patterns):
            self.weights += np.outer(pat,pat)
            np.fill_diagonal(self.weights,1)
        return self

    def recall(self,pat):
        p = bool2pm(pat)
        for i in range(25):
            r = np.sign(np.dot(self.weights,p))
            if closeEnough(p,r,.99): # replace with a threshold for differences
                print(np.sum(r))
                return r
            p = r
        print(np.sum(p))
        return pm2bool(p)


class HopfieldNetwork:

    def __init__(self,size):
        self.size = size
        self.weights = np.zeros((size,size),dtype=int)

    def train(self,patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern,pattern).astype(int)
            np.fill_diagonal(self.weights,1)
        return self

    def recall(self,pattern):
        for i in range(25):
            old_pattern = np.copy(pattern)
            pattern = np.sign(np.dot(self.weights,pattern).astype(int))
            if closeEnough(pattern,old_pattern,.99): # replace with earthmovers
            #if np.array_equal(pattern,old_pattern): # replace with earthmovers
                #print('returning early ;-) ... at %i'%i)
                print(np.sum(pattern))
                return pattern
        print(np.sum(pattern))
        return pattern


        

