#!/usr/bin/python3

import Molecules
import matplotlib.pyplot as plt

def main():
    m = [Molecules.Molecule() for i in range(5)]
    m[0].setParams([5,50,150],[1,4,4],[50,10,20]).setPDF()
    m[1].setParams([8,70,120],[1,4,4],[40,20,30]).setPDF()
    m[2].setParams([15,140,200],[1,4,4],[60,15,40]).setPDF()
    m[3].setParams([10,130],[1,4],[45,22]).setPDF()
    m[4].setParams([13,220],[1,4],[55,40]).setPDF()
    '''
    for mol in m:
        print('#################')
        mol.printPDF()

    print(m[4].samplePDF(10))
    '''
    plt.plot(m[-1].sampleBINvec(10))
    plt.show()



if __name__ == "__main__":
    main()
