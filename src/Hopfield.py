import numpy as np
import matplotlib.pyplot as plt
import sys

from utils import closeEnough,bool2pm,pm2bool,energy

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

class HopfieldNetwork_theta:
    def __init__(self,sz):
        self.sz = sz
        self.weights = np.zeros((sz,sz),dtype=float)
        self.theta = []
        self.max_iterations = 500

    def train(self,patterns):
        nP,nN = patterns.shape
        print(self.weights.shape)
        assert nN==self.sz
        for pattern in patterns:
            self.weights += np.outer(pattern,pattern)
            np.fill_diagonal(self.weights,self.sz)
        self.weights /= nP
        self.theta = -np.mean(self.weights,axis=1)
        return self

    def recall(self,pattern):
        s = np.copy(pattern)
        for i in range(self.max_iterations):
            olden = energy(self.weights,s,self.theta)
            news = np.sign(np.inner(self.weights,s) + self.theta)
            newen = energy(self.weights,news,self.theta)
            if closeEnough(olden,newen):
                #print(olden,newen)
                break
            else:
                s = np.copy(news)
        print('%i\t%f'%(i,np.sum(s)))
        return s

    def set_max_iter(self,x):
        self.max_iterations = int(x)
        return self
        
        
        

