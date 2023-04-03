#!/sdf/sw/images/slac-ml/20211027.1/bin/python3
import numpy as np


class HopfieldNetwork:
        def __init__(self,size):
            self.size = size
            self.weights = np.zeros((size,size))
        def train(self,patterns):
            for pattern in patterns:
                pattern = np.reshape(pattern,(self.size,1))
                self.weights += np.dot(pattern,pattern.T)
                np.fill_diagonal(self.weights,1)
                # print(self.weights)
        def recall(self,patterns):
            pattern = np.reshape(patterns,(self.size,1))
            for i in range(25):
                old_pattern = np.copy(pattern)
                #pattern = np.dot(self.weights,pattern)
                pattern = np.sign(np.dot(self.weights,pattern))
                if np.array_equal(pattern,old_pattern): # replace with earthmovers
                    print('returning early ;-) ... at %i'%i)
                    return pattern.flatten()
            return pattern.flatten()


def Thresh(npatterns,size,ratio=0.5):
    rando = np.random.random(size=(npatterns,size))
    out = np.zeros((npatterns,size))
    inds = np.where(rando>ratio)
    out[inds] = 2
    out -= 1
    return out

def distort(npatterns,orig,ratio=0.1):
    out = np.zeros((npatterns,orig.shape[1]))
    sampleinds = np.random.choice(np.arange(orig.shape[0]),npatterns)
    for i,s in enumerate(sampleinds):
        out[i,:] = np.copy(orig[s,:])
    
    for p in range(out.shape[0]):
        for v in range(out.shape[1]):
            if np.random.random()<ratio:
                if np.sign(out[p,v])==-1:
                    out[p,v] = 1
                else:
                    out[p,v] = -1
    return out
        
def Perturb(x, p=0.1):
    '''
    y = Perturb(x, p=0.1)
               
    Apply binary noise to x. With probability p, each bit will be randomly
    set to -1 or 1.
                                               
    Inputs:
    x is an array of binary vectors of {-1,1}
    p is the probability of each bit being randomly flipped
                                                                                       
    Output:
    y is an array of binary vectors of {-1,1}
    '''
    y = np.copy(x)
    for yy in y:
        for k in range(len(yy)):
            if np.random.random()<p:
                yy[k] *= np.sign(yy[k])
    return y


def main():
    N = 5 #num neurons
    n = 5 #num patterns
    patterns = Thresh(n,N,ratio=.5)
    print('orig:',patterns)
    orig = np.copy(patterns)

    network = HopfieldNetwork(N)
    network.train(patterns)
    print('final',patterns)

    testn=7
    newpatterns = distort(testn,orig,ratio=.5)
    for newp in newpatterns:
        print('newtest',newp)
        print('newfinal:',network.recall(newp))



if __name__ == '__main__':
    main()
